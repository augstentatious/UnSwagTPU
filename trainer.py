import jax
import jax.numpy as jnp
from sophia_pallas.core import sophia_forward
import optax

@jax.jit
def mse_loss(A, B, x, W, target):
    """Calculates structural divergence (Error)."""
    pred = sophia_forward(x, W, A, B)
    return jnp.mean((pred - target) ** 2)

@jax.jit
def train_step(A, B, x, W, target, learning_rate=0.01):
    """
    Pure XLA Update Step.
    Only computes gradients for A and B. W is ignored.
    """
    # Calculate gradients
    loss_val, (grads_A, grads_B) = jax.value_and_grad(mse_loss, argnums=(0, 1))(A, B, x, W, target)
    
    # Simple SGD Update (Can be swapped for Adam/Lion later)
    new_A = A - learning_rate * grads_A
    new_B = B - learning_rate * grads_B
    
    return loss_val, new_A, new_B