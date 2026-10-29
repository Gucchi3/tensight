# @file lsq.py
# @brief LSQ fake quantization and its gradient scale
# @author Kohei Yamaguchi, Claude

import math
import torch

from ..formats import NumberFormat, RoundMode, format_quantize



def lsq_quantize(x: torch.Tensor, step: torch.Tensor, zero_point: torch.Tensor, number_format: NumberFormat, mode: RoundMode, grad_scale: float) -> torch.Tensor:
    '''
    #* Fake-quantize x with a learned step (LSQ) and a learned zero point (LSQ+).
    #? Autograd gives the LSQ gradients: d/dstep = round(v) - v inside the range, and the end of the range outside.
    #! The codes are rounded before the zero point is added, like encode(). Adding the zero point first in float32 could move a value next to a tie onto it.
    '''
    
    # Divide by the step, and put the LSQ gradient scale on the step
    step   = lsq_scale_gradient(step, grad_scale)
    scaled = x / step
    
    # Round (then add the zero point and clip) for the value, and take the gradients from the clipped value with a straight-through path
    clipped = (scaled + zero_point).clamp(number_format.min_value, number_format.max_value)
    grid    = format_quantize(scaled, number_format, mode, zero_point).detach() + (clipped - clipped.detach())
    return (grid - zero_point) * step


def lsq_grad_scale(count: int, number_format: NumberFormat) -> float:
    '''
    #* Return the LSQ gradient scale 1 / sqrt(N x Qp).
    #? N is the number of values that share one step. Qp is the largest value of the format.
    '''
    
    return 1.0 / math.sqrt(max(count, 1) * number_format.max_value)


def lsq_scale_gradient(x: torch.Tensor, factor: float) -> torch.Tensor:
    '''
    #* Keep the value of x exactly, and multiply only its gradient by factor.
    '''
    
    scaled = x * factor
    return x.detach() + (scaled - scaled.detach())
