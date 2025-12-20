# unswag/kernels/__init__.py

# default to None
triton_ops = None
tpu_ops = None

# 1. Try Loading GPU (Triton) Kernels
try:
    from . import triton_ops
except ImportError:
    # This happens if 'triton' isn't installed or no GPU is found.
    # We suppress the error so the TPU/CPU path can still run.
    pass

# 2. Try Loading TPU (Pallas/Mosaic) Kernels (Future Proofing)
try:
    # from . import tpu_ops 
    pass
except ImportError:
    pass
