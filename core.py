import jax
import jax.numpy as jnp

def unswag_compress_jax(x):
    """
    Pure JAX compression: 32x, TPU-friendly, handles any shape.
    """
    x_flat = jnp.asarray(x).flatten()
    N_original = x_flat.shape[0]
    
    # Quantize to 1-bit (Sign)
    bits = (x_flat > 0).astype(jnp.uint8)
    
    # Pad to multiple of 8 for packing
    pad_bits = 8 - (N_original % 8)
    if pad_bits != 8:
        bits = jnp.pad(bits, (0, pad_bits), constant_values=0)
    
    # Pack bits (Fuseable by XLA)
    packed = jnp.packbits(bits.reshape(-1, 8), axis=1, bitorder='little').flatten()
    
    # Return compressed data + metadata
    return packed, x.shape, N_original, pad_bits

def unswag_decompress_jax(packed, original_shape, original_N, pad_bits):
    """
    Pure JAX decompression: Handles dynamic shapes correctly.
    """
    # Unpack bits
    bits_unpacked = jnp.unpackbits(packed.reshape(-1, 1), axis=1, bitorder='little')
    bits_unpacked = bits_unpacked[:, :8].flatten() 
    
    # Remove padding using static slicing
    if pad_bits != 8:
        bits_unpacked = bits_unpacked[:-pad_bits]
    
    # Resurrect to Float (Isomorphic Sign Restoration)
    # 1 -> 1.0, 0 -> 0.0 (Perfect for ReLU)
    restored = bits_unpacked.astype(jnp.float32)
    
    return restored.reshape(original_shape)

class UnSwagActivations:
    """Core wrapper interface."""
    
    @staticmethod
    def compress(activation):
        packed, shape, N, pad = unswag_compress_jax(activation)
        return {
            'compressed_data': packed,
            'metadata': (shape, N, pad)
        }
    
    @staticmethod
    def restore(checkpoint_obj):
        compressed = checkpoint_obj['compressed_data']
        shape, N, pad = checkpoint_obj['metadata']
        return unswag_decompress_jax(compressed, shape, N, pad)
