import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .components import DifferentiableRouter, HardwareOptimizedTether, AdaptiveSummaryRegister

class SparseGlobalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_dim = config.hidden_dim
        self.head_dim = config.hidden_dim // config.num_heads
        
        # Guard against indivisible dims
        assert config.hidden_dim % config.num_heads == 0, f"Dim {config.hidden_dim} not divisible by {config.num_heads} heads"
        
        self.scale = math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.reg_k = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.reg_v = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

    def forward(self, x, mask, register):
        B, L, D = x.shape
        batch_indices = [torch.where(mask[b])[0] for b in range(B)]
        max_k = max([len(idx) for idx in batch_indices])
        if max_k == 0: return x
        
        signal_tokens = torch.zeros(B, max_k, D, device=x.device)
        pos = torch.zeros(B, max_k, dtype=torch.long, device=x.device)
        for b, idx in enumerate(batch_indices):
            k = len(idx)
            signal_tokens[b, :k] = x[b, idx]
            pos[b, :k] = idx

        # Explicitly use self.head_dim to avoid inference errors
        Q = self.q_proj(signal_tokens).view(B, max_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        K_tokens = self.k_proj(signal_tokens)
        V_tokens = self.v_proj(signal_tokens)
        K_reg = self.reg_k(register).unsqueeze(1)
        V_reg = self.reg_v(register).unsqueeze(1)
        
        K = torch.cat([K_tokens, K_reg], dim=1).view(B, max_k + 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = torch.cat([V_tokens, V_reg], dim=1).view(B, max_k + 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Simple causal mask for the sparse indices
        q_pos = pos.unsqueeze(2) 
        k_pos = pos.unsqueeze(1)
        mask_val = (q_pos < k_pos).float() * -1e9
        reg_mask = torch.zeros(B, max_k, 1, device=x.device)
        full_mask = torch.cat([mask_val, reg_mask], dim=2).unsqueeze(1)
        
        attn = F.softmax(scores + full_mask, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, max_k, D)
        
        res = x.clone()
        for b, idx in enumerate(batch_indices):
            res[b, idx] = self.out_proj(out)[b, :len(idx)]
        return res

class ProtocolCLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = DifferentiableRouter(config)
        self.local_cnn = HardwareOptimizedTether(config)
        self.register_updater = AdaptiveSummaryRegister(config)
        self.sparse_attention = SparseGlobalAttention(config)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim)
        )
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)

    def forward(self, x, register):
        packets, conf = self.router(x)
        
        # Soft-routing for training gradients
        p01 = conf[:, :, 1].unsqueeze(-1)
        p11 = conf[:, :, 3].unsqueeze(-1)
        
        cnn_out = self.local_cnn(x)
        attn_out = self.sparse_attention(x, (packets == 3), register)
        
        out = (p01 * cnn_out) + (p11 * attn_out) + ((1 - p01 - p11) * x)
        
        out = self.norm1(out + x)
        out = self.norm2(out + self.ffn(out))
        
        return out, self.register_updater(x, (packets == 2), register), conf

class PSAStack(nn.Module):
    def __init__(self, config, layers=12):
        super().__init__()
        self.layers = nn.ModuleList([ProtocolCLayer(config) for _ in range(layers)])
        
    def forward(self, x, reg=None):
        B, L, D = x.shape
        if reg is None:
            reg = torch.zeros(B, D, device=x.device)
        
        for layer in self.layers:
            x, reg, _ = layer(x, reg)
        return x, reg
