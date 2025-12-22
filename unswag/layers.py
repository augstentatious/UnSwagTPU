import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
from .kernels.triton_ops import (
    _pack_2bit_silu_kernel, 
    _unpack_2bit_backward_kernel, 
    pack_activations, 
    unpack_activations
)

class UnSwagSiLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        n_elements = x.numel()
        out_size = (n_elements + 3) // 4
        packed_mask = torch.empty(out_size, dtype=torch.int8, device=x.device)
        grid = lambda meta: (triton.cdiv(n_elements // 4, meta['BLOCK_SIZE']),)
        _pack_2bit_silu_kernel[grid](x, packed_mask, n_elements, BLOCK_SIZE=1024)
        ctx.save_for_backward(packed_mask)
        ctx.n_elements = n_elements
        return F.silu(x)

    @staticmethod
    def backward(ctx, grad_output):
        packed_mask, = ctx.saved_tensors
        n_elements = ctx.n_elements
        grad_input = torch.empty_like(grad_output)
        grid = lambda meta: (triton.cdiv(n_elements // 4, meta['BLOCK_SIZE']),)
        _unpack_2bit_backward_kernel[grid](
            grad_output, packed_mask, grad_input, n_elements, BLOCK_SIZE=1024
        )
        return grad_input

class UnSwagSiLU(nn.Module):
    def forward(self, x):
        return UnSwagSiLUFunction.apply(x)

class UnSwagLinear(nn.Linear):
    def forward(self, x):
        packed_x = pack_activations(x)
        x_quantized = unpack_activations(packed_x, x.shape, x.dtype)
        return F.linear(x_quantized, self.weight, self.bias)
