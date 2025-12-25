from dataclasses import dataclass

@dataclass
class PSAConfig:
    hidden_dim: int = 768
    num_heads: int = 12
    conv_kernel_size: int = 5
    null_confidence_threshold: float = 0.99
    anchor_ema_alpha: float = 0.1
    dropout: float = 0.1
