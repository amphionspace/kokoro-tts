import random

import torch
from torch import nn
from torch.nn import functional as F

from training.common import STYLE_SRC
from utils import length_to_mask, mask_from_lens, maximum_path, log_norm


class Generator(nn.Module):
    """Single DDP boundary includes every path to the predictor and decoder."""
    def __init__(self, model, stage, max_mel_frames=400, voicepack_fraction=0.5):
        super().__init__()
        self.nets = nn.ModuleDict({k:v for k,v in model.items() if k not in ('mpd','msd','wd')})
        self.stage = stage
        self.max_mel_frames = max_mel_frames
        self.voicepack_fraction = voicepack_fraction

    def set_mode(self, train=True, joint=False):
        self.train(train)
        active = (['text_aligner','text_encoder','style_encoder','decoder'] if self.stage == 1
                  else ['bert','bert_encoder','predictor','predictor_encoder'] + (['style_encoder','decoder'] if joint else []))
        if self.stage == 2 and joint and 'voicepack' in self.nets and bool(self.nets['voicepack'].initialized):
            active.append('voicepack')
        for key, module in self.nets.items():
            module.train(train and key in active)
            module.requires_grad_(key in active)
        return active

    def forward(self, batch, deterministic=False):
        net = self.nets
        texts, lengths = batch['tokens'], batch['token_lengths']
        mels, mel_lengths = batch['mels'], batch['mel_lengths']
        text_mask = length_to_mask(lengths)
        mel_mask = length_to_mask(mel_lengths // 2)
        with torch.set_grad_enabled(self.training and self.stage == 1):
            _, logits, attn = net['text_aligner'](mels, mel_mask, texts)
            attn = attn.transpose(-1,-2)[...,1:].transpose(-1,-2)
            valid = (~text_mask).unsqueeze(-1) & (~mel_mask).unsqueeze(1)
            attn = attn.masked_fill(~valid,0)
            with torch.no_grad():
                mono = maximum_path(attn.float(), mask_from_lens(attn,lengths,mel_lengths//2))
            # Packed cuDNN LSTM backward is unstable under BF16 on these batches.
            with torch.autocast('cuda', enabled=False):
                encoded = net['text_encoder'](texts,lengths,text_mask)
            alignment = attn if self.stage == 1 and self.training and not deterministic and random.random() < 0.5 else mono
            aligned = encoded @ alignment
        # One shared assignment for duration, F0/energy and decoder conditioning.
        voice_mask = None
        if self.stage == 2 and self.training and 'voicepack' in net and net['voicepack'].vector.requires_grad:
            count = max(1, min(len(texts)-1, round(len(texts)*self.voicepack_fraction))) if len(texts)>1 else 1
            voice_mask = torch.zeros(len(texts), 1, dtype=torch.bool, device=texts.device)
            voice_mask[torch.randperm(len(texts), device=texts.device)[:count]] = True
        losses = {}
        if self.stage == 1:
            losses['s2s'] = torch.stack([F.cross_entropy(p[:n],t[:n]) for p,t,n in zip(logits,texts,lengths)]).mean()
            losses['mono'] = (attn-mono).abs().masked_select(valid).mean()*10
        else:
            styles = torch.cat([net['predictor_encoder'](mels[i:i+1,:,:n].unsqueeze(1)) for i,n in enumerate(mel_lengths)])
            if voice_mask is not None:
                styles = torch.where(voice_mask, net['voicepack'].vector[:,128:], styles)
            bert = net['bert'](texts,attention_mask=(~text_mask).int())
            d_en = net['bert_encoder'](bert).transpose(-1,-2)
            duration_logits, prosody = net['predictor'](d_en,styles,lengths,mono,text_mask)
            d_gt = mono.sum(-1).detach()
            duration_losses, ce_losses = [], []
            for pred, target, length in zip(duration_logits,d_gt,lengths):
                pred,target = pred[:length], target[:length]
                binary = torch.arange(pred.shape[-1],device=pred.device)[None,:] < target[:,None]
                duration_losses.append(F.l1_loss(pred.sigmoid().sum(-1)[1:-1],target[1:-1]))
                ce_losses.append(F.binary_cross_entropy_with_logits(pred,binary.float()))
            losses['duration'] = torch.stack(duration_losses).mean()
            losses['duration_ce'] = torch.stack(ce_losses).mean()
        clip_frames = min(int(mel_lengths.min())//2-1,self.max_mel_frames//2)
        if clip_frames < 40:
            raise ValueError('Too short for style encoder; should be excluded by preflight')
        chunks, targets, waves, p_chunks = [],[],[],[]
        for i,length in enumerate(mel_lengths):
            maximum = int(length)//2-clip_frames
            start = maximum//2 if deterministic else random.randrange(maximum)
            chunks.append(aligned[i,:,start:start+clip_frames])
            targets.append(mels[i,:,start*2:(start+clip_frames)*2])
            waves.append(batch['waves'][i][start*600:(start+clip_frames)*600])
            if self.stage == 2:
                p_chunks.append(prosody[i,:,start:start+clip_frames])
        asr = torch.stack(chunks)
        mel = torch.stack(targets).detach()
        target_wave = torch.stack(waves).to(mels.device)
        with torch.no_grad():
            f0, _, _ = net['pitch_extractor'](mel.unsqueeze(1))
            energy = log_norm(mel.unsqueeze(1)).squeeze(1)
        acoustic_style = net['style_encoder'](mel.unsqueeze(1))
        if self.stage == 2:
            prosodic_style = net['predictor_encoder'](mel.unsqueeze(1))
            if voice_mask is not None:
                acoustic_style = torch.where(voice_mask, net['voicepack'].vector[:,:128], acoustic_style)
                prosodic_style = torch.where(voice_mask, net['voicepack'].vector[:,128:], prosodic_style)
            predicted_f0, predicted_energy = net['predictor'].F0Ntrain(torch.stack(p_chunks),prosodic_style)
            losses['f0'] = F.smooth_l1_loss(predicted_f0,f0)/10
            losses['energy'] = F.smooth_l1_loss(predicted_energy,energy)
            f0,energy = predicted_f0,predicted_energy
        wave = net['decoder'](asr,f0,energy,acoustic_style).squeeze(1)
        if wave.shape != target_wave.shape:
            raise ValueError(f'Waveform alignment mismatch: {wave.shape}, {target_wave.shape}')
        return wave,target_wave,losses


class Discriminators(nn.Module):
    def __init__(self, model):
        super().__init__()
        from losses import DiscriminatorLoss, GeneratorLoss
        self.mpd, self.msd = model.mpd, model.msd
        self.d_loss = DiscriminatorLoss(self.mpd,self.msd)
        self.g_loss = GeneratorLoss(self.mpd,self.msd)

    def forward(self, target, generated):
        return self.d_loss(target[:,None,:],generated[:,None,:])

    def generator_loss(self, target, generated):
        return self.g_loss(target[:,None,:],generated[:,None,:])
