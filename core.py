import jax
import jax.numpy as jnp
from jax import random

def sophia_forward(x, W, A, B):
    """
    The Sophia Protocol Forward Pass.
    Implements Low-Rank Adaptation (LoRA) injection efficiently in JAX.
    
    Args:
        x: Input tensor (Batch, Dim)
        W: Frozen Base Weights (Dim, Hidden)
        A: Adapter Down-Projection (Dim, Rank)
        B: Adapter Up-Projection (Rank, Hidden)
    """
    # 1. Frozen Path (The Base Model)
    h_frozen = jnp.dot(x, W)
    
    # 2. Structural Path (The Wisdom)
    # B @ (A @ x) - computed sequentially for memory efficiency
    h_adapter = jnp.dot(jnp.dot(x, A), B)
    
    # 3. Structural Isomorphism (Sum)
    return h_frozen + h_adapter

def init_sophia_weights(key, dim, hidden, rank=16):
    """Initializes the Adapter weights (A, B) using Kaiming/Xavier logic."""
    k1, k2 = random.split(key)
    # A initializes near zero or small noise
    A = random.normal(k1, (dim, rank)) * 0.01
    # B initializes to zero (standard LoRA practice to start as identity)
    B = jnp.zeros((rank, hidden))
    return A, B