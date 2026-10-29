# @file ste.py
# @brief Straight-through gradients (STE) for rounding and quantization
# @author Kohei Yamaguchi, Claude

import torch

from .format import format_quantize
from .round  import build_round
from .types  import NumberFormat, RoundMode



def ste_round(x: torch.Tensor, mode: RoundMode) -> torch.Tensor:
    '''
    #* Round x in the forward pass, and pass the gradient as it is in the backward pass.
    '''
    
    return build_round(mode)(x).detach() + (x - x.detach())


def ste_quantize(x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, number_format: NumberFormat, mode: RoundMode) -> torch.Tensor:
    '''
    #* Fake-quantize x with the scale and the zero point. The gradient passes only inside the range.
    #? Quantizer uses this for scalers that do not learn.
    #! The scale and the zero point get no gradient.
    '''
    
    # Round onto the grid of the format, then add the zero point
    scale      = scale.detach()
    zero_point = zero_point.detach()
    scaled     = x / scale
    grid       = format_quantize(scaled, number_format, mode, zero_point)
    inside     = (scaled + zero_point >= number_format.min_value) & (scaled + zero_point <= number_format.max_value)
    
    # Go back to real values, and pass the gradient straight through
    return ste_attach(x, (grid - zero_point) * scale, inside)


def ste_attach(reference: torch.Tensor, value: torch.Tensor, inside: torch.Tensor | None = None) -> torch.Tensor:
    '''
    #* Return value in the forward pass, and use the gradient of reference in the backward pass.
    #? The fake ops use this to give gradients to the exact results of the integer math.
    #! Where inside is False, the gradient is zero. The forward value is exactly value (no float error).
    '''
    
    attached = value.detach() + (reference - reference.detach())
    if inside is None:
        return attached
    return torch.where(inside, attached, value.detach())
