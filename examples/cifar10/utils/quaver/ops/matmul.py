# @file matmul.py
# @brief Matrix product of two quantized tensors (for attention)
# @author Kohei Yamaguchi, Claude

import torch

from ..formats   import ste_attach
from ..scales    import QuantParams, params_codes, params_math_dtype, params_values
from .accumulate import ACCUMULATOR_FORMAT, accumulate_matmul, accumulate_product



def matmul_fake(a: torch.Tensor, a_params: QuantParams, b: torch.Tensor, b_params: QuantParams) -> tuple[torch.Tensor, QuantParams]:
    '''
    #* Training mode matrix product: the exact integer accumulator times its scale, with float gradients.
    #? The inputs must be fake-quantized values. FP inputs become integers on the way (FP4 values are doubled).
    #! The codes are taken and turned back into values in the type of params_math_dtype(). The accumulator itself is always exact (float64).
    '''
    
    dtype       = params_math_dtype()
    acc, params = matmul_integer(params_codes(a, a_params, dtype), a_params, params_codes(b, b_params, dtype), b_params)
    return ste_attach(accumulate_product(a, b), params_values(acc, params, dtype)), params


def matmul_integer(q_a: torch.Tensor, a_params: QuantParams, q_b: torch.Tensor, b_params: QuantParams) -> tuple[torch.Tensor, QuantParams]:
    '''
    #* Integer mode matrix product: the exact accumulator (int64) and its parameters.
    #? The scale of the accumulator is (code scale of a) x (code scale of b), so scales for each row and each column broadcast.
    '''
    
    acc   = accumulate_matmul(q_a, a_params, q_b, b_params)
    scale = a_params.code_scale * b_params.code_scale
    return acc, QuantParams(scale, torch.zeros([1] * scale.ndim, device=scale.device), ACCUMULATOR_FORMAT)


def matmul_float(a: torch.Tensor, a_params: QuantParams, b: torch.Tensor, b_params: QuantParams) -> tuple[torch.Tensor, QuantParams]:
    '''
    #* Float mode (calibration): the float matrix product, and the parameters that the accumulator has in the other modes.
    '''
    
    scale = a_params.code_scale * b_params.code_scale
    return accumulate_product(a, b), QuantParams(scale, torch.zeros_like(scale), ACCUMULATOR_FORMAT)
