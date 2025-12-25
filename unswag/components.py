import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import PSAConfig

class DifferentiableRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 4)
        )
    def forward(self, x, temperature=1.0):
        logits = self.classifier(x)
        if self.training:
            soft_packets = F.gumbel_softmax(logits, tau=temperature, hard=True)
            packets = torch.argmax(soft_packets, dim=-1)
        else:
            packets = torch.argmax(logits, dim=-1)
        return packets, F.softmax(logits, dim=-1)

class HardwareOptimizedTether(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.depthwise = nn.Conv1d(config.hidden_dim, config.hidden_dim, 
                                   kernel_size=config.conv_kernel_size, 
                                   padding=config.conv_kernel_size // 2, 
                                   groups=config.hidden_dim, bias=False)
        self.pointwise = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.norm = nn.LayerNorm(config.hidden_dim)
    def forward(self, x):
        feat = x.transpose(1, 2)
        feat = self.depthwise(feat).transpose(1, 2)
        return self.norm(self.pointwise(F.gelu(feat)))

class AdaptiveSummaryRegister(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(config.anchor_ema_alpha))
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.norm = nn.LayerNorm(config.hidden_dim)
    def forward(self, x, mask, register):
        if not mask.any(): return register
        projected = self.proj(x)
        weights = mask.float().unsqueeze(-1)
        anchor_mean = (projected * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-6)
        return self.norm(register + torch.sigmoid(self.alpha) * (anchor_mean - register))
