"""The pinned 7.48M network with strict loading and teacher-duration training."""
import json
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AlbertConfig
from training.common import ROOT, CODE_ROOT, BASE_KEYS, load_exact
from kokoro7m.modules import CustomAlbert,ProsodyPredictor,TextEncoder
from kokoro7m.istftnet import Decoder


def sample_window_start(frames, window, deterministic=False):
    """Uniform random crop; no special weighting for sentence boundaries."""
    high = frames - window
    if high < 0:
        raise ValueError('Window exceeds sentence')
    return high // 2 if deterministic else int(torch.randint(high + 1, ()))


class Student(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        c=config or json.loads((CODE_ROOT/'configs/student_7m.json').read_text());self.config=c
        self.bert=CustomAlbert(AlbertConfig(vocab_size=c['n_token'],**c['plbert']))
        self.bert_encoder=nn.Linear(c['plbert']['hidden_size'],c['hidden_dim'])
        self.predictor=ProsodyPredictor(style_dim=c['style_dim'],d_hid=c['hidden_dim'],nlayers=c['n_layer'],max_dur=c['max_dur'],dropout=c['dropout'])
        self.text_encoder=TextEncoder(channels=c['hidden_dim'],kernel_size=c['text_encoder_kernel_size'],depth=c['n_layer'],n_symbols=c['n_token'])
        self.decoder=Decoder(dim_in=c['hidden_dim'],style_dim=c['style_dim'],dim_out=c['n_mels'],disable_complex=True,**c['istftnet'])

    def load_nested(self,path):
        weights=torch.load(path,map_location='cpu',weights_only=True)
        if set(weights)!=set(BASE_KEYS):raise ValueError(f'Unexpected model groups: {list(weights)}')
        for key in BASE_KEYS:load_exact(getattr(self,key),weights[key])
        return {'parameters':sum(p.numel() for p in self.parameters()),'loaded_parameters':sum(p.numel() for p in self.parameters()),'strict':True}

    def export(self,path):
        torch.save({key:getattr(self,key).state_dict() for key in BASE_KEYS},path)

    def encode(self,ids,lengths,style):
        mask=torch.arange(ids.shape[1],device=ids.device)[None]>=lengths[:,None]
        # Packed recurrent kernels remain FP32, including under outer BF16.
        with torch.autocast('cuda',enabled=False):
            contextual=self.bert_encoder(self.bert(ids,attention_mask=(~mask).int())).transpose(-1,-2)
            d=self.predictor.text_encoder(contextual.float(),style[:,128:].float(),lengths,mask)
            packed=nn.utils.rnn.pack_padded_sequence(d,lengths.cpu(),batch_first=True,enforce_sorted=False)
            x,_=self.predictor.lstm(packed)
            x,_=nn.utils.rnn.pad_packed_sequence(x,batch_first=True,total_length=ids.shape[1])
            duration=torch.sigmoid(self.predictor.duration_proj(x)).sum(-1)
            content=self.text_encoder(ids,lengths,mask)
        return d,content,duration,mask

    def aligned(self,d,content,durations,lengths):
        frames=durations.sum(-1).long();maximum=int(frames.max())
        alignment=content.new_zeros((len(lengths),durations.shape[1],maximum))
        for i in range(len(lengths)):
            index=torch.repeat_interleave(torch.arange(int(lengths[i]),device=d.device),durations[i,:lengths[i]].long())
            alignment[i,index,torch.arange(len(index),device=d.device)]=1
        return d.transpose(-1,-2)@alignment,content@alignment,frames

    def forward(self,batch,style,crop_frames=120,context_frames=0,deterministic=False,bf16=True):
        ids,lengths,target_dur=batch['ids'],batch['lengths'],batch['durations']
        style=style.expand(len(lengths),-1)
        d,content,pred_dur,mask=self.encode(ids,lengths,style)
        prosody,asr,frames=self.aligned(d,content,target_dur,lengths)
        with torch.autocast('cuda',enabled=False):
            f0,energy=self.predictor.F0Ntrain(prosody.float(),style[:,128:].float())
        valid=torch.arange(f0.shape[1],device=f0.device)[None]<frames[:,None]*2
        losses={'duration':((pred_dur-target_dur).abs()*(~mask)).sum()/(~mask).sum(),
                'f0':(F.smooth_l1_loss(f0,batch['f0'],reduction='none')*valid).sum()/valid.sum()/10,
                'energy':((energy-batch['energy']).abs()*valid).sum()/valid.sum()}
        # Match fine-tuning: score the complete decoded window, including edges.
        # Uniform random windows, without special boundary oversampling.
        if context_frames != 0:
            raise ValueError('Full-window supervision requires context_frames=0')
        window=min(crop_frames,int(frames.min()))
        inputs=[];pitches=[];energies=[];targets=[]
        for i,frames_i in enumerate(frames.tolist()):
            start=sample_window_start(frames_i,window,deterministic)
            inputs.append(asr[i,:,start:start+window]);pitches.append(f0[i,start*2:(start+window)*2]);energies.append(energy[i,start*2:(start+window)*2])
            targets.append(batch['audio'][i][start*600:(start+window)*600])
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16):
            wave=self.decoder(torch.stack(inputs),torch.stack(pitches),torch.stack(energies),style[:,:128]).squeeze(1).float()
        target=torch.stack(targets).to(wave.device)
        if wave.shape!=target.shape:raise ValueError((wave.shape,target.shape))
        return wave,target,losses

    @torch.inference_mode()
    def synthesize(self,token_ids,style):
        ids=torch.tensor([[0,*token_ids,0]],device=style.device)
        lengths=torch.tensor([ids.shape[1]],device=style.device)
        if ids.shape[1]>512:raise ValueError('Student input exceeds 510 effective tokens')
        d,content,dur,_=self.encode(ids,lengths,style)
        dur=dur.round().clamp(min=1).long()
        if int(dur.sum())>4800:raise ValueError('Predicted student audio exceeds 120 seconds')
        prosody,asr,frames=self.aligned(d,content,dur,lengths)
        f0,energy=self.predictor.F0Ntrain(prosody,style[:,128:])
        wave=self.decoder(asr,f0,energy,style[:,:128]).reshape(-1)
        assert wave.numel()==int(frames.sum())*600
        return wave,dur[0]
