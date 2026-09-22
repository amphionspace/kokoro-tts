"""Magnitude, silence, and perceptual objectives; GAN lives in the shared vendor."""
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F

class AcousticLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel=torchaudio.transforms.MelSpectrogram(sample_rate=24000,n_fft=1024,hop_length=256,n_mels=80,power=1.0)
        for window in [600,1200,240]:self.register_buffer('window_'+str(window),torch.hann_window(window))
    def forward(self,pred,target):
        spectral=pred.new_zeros(())
        for nfft,hop,win in [(1024,120,600),(2048,240,1200),(512,50,240)]:
            args=dict(n_fft=nfft,hop_length=hop,win_length=win,window=getattr(self,'window_'+str(win)),return_complex=True)
            a=torch.stft(pred,**args).abs().clamp_min(1e-5);b=torch.stft(target,**args).abs().clamp_min(1e-5)
            convergence=(a-b).flatten(1).norm(dim=1)/b.flatten(1).norm(dim=1).clamp_min(1e-5)
            spectral=spectral+convergence.mean()+F.l1_loss(a.log(),b.log())
        mel=F.l1_loss((self.mel(pred)+1e-5).log(),(self.mel(target)+1e-5).log())
        n=pred.shape[-1]//240
        a=(pred[:,:n*240].reshape(-1,n,240).square().mean(-1)+1e-12).sqrt()
        b=(target[:,:n*240].reshape(-1,n,240).square().mean(-1)+1e-12).sqrt()
        silent=b<1e-3;silence=(F.relu(a-b)*silent).sum()/silent.sum().clamp_min(1)
        return {'stft':spectral/3,'mel':mel,'silence':silence,'silent_fraction':silent.float().mean()}
