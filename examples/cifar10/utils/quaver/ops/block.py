# @file block.py
# @brief Quantize to block-scaled formats (NVFP4 and the like) like the device: the block scales from the block maxima, then the elements, with integers only
# @author Kohei Yamaguchi, Claude

import torch

from ..formats   import NumberFormat, build_format, device_define, format_to_integer
from ..scales    import BlockScale, Granularity, GranularityKind, QuantParams, compute_block, group_merge, group_split, params_attach, params_codes, params_math_dtype, params_note
from .requantize import requantize_params
from .wide       import wide_round_format



def block_quantize_fake(x: torch.Tensor, x_params: QuantParams, number_format: NumberFormat, scale_format: NumberFormat, tensor_scale: torch.Tensor, granularity: Granularity, style_name: str) -> tuple[torch.Tensor, QuantParams]:
    '''
    #* Training mode: quantize to the block-scaled format with the integer math of block_quantize_integer, with straight-through gradients.
    #? x must hold exact values of codes with one scale for the tensor (the output of another op), so turn off TF32.
    #! The gradient passes where the value fits under the top of its block (the top of the format x its block scale).
    '''
    
    dtype         = params_math_dtype()
    codes, params = block_quantize_integer(params_codes(x, x_params, dtype), x_params, number_format, scale_format, tensor_scale, granularity, style_name)
    inside        = x.abs() <= number_format.max_value * params.scale
    return params_attach(x, codes, params, inside, dtype), params


def block_quantize_integer(q: torch.Tensor, q_params: QuantParams, number_format: NumberFormat, scale_format: NumberFormat, tensor_scale: torch.Tensor, granularity: Granularity, style_name: str) -> tuple[torch.Tensor, QuantParams]:
    '''
    #* Integer mode: quantize codes with one static scale (codes of a format, or an accumulator of any width) to a block-scaled format, as the device does it while running.
    #? Block scale code = |largest code of the block| x M6 / 2^n6, rounded half to even onto scale_format. Element code = code x Mr / (block scale code x 2^nr), rounded half to even.
    #? On the device the division by the block code can be comparisons with the midpoints of the grid times the block code. The device op torch.ops.quaver.block_quantize does it, with M6, n6, Mr, and nr from here.
    #! q_params must have one scale for the tensor. Integers only, exact for any width of q (wide_round_format).
    '''
    
    # The static ratios of one step of q to one step of the block scales, and to one step of the elements (counted in steps of the block scales)
    step     = tensor_scale.double() * scale_format.step_min
    to_scale = requantize_params(q_params.code_scale.double() / (number_format.max_value * step), style_name)
    to_value = requantize_params(q_params.code_scale.double() / (number_format.step_min * step) , style_name)
    
    # The codes of the elements and of their block scales, and the scale of each value
    codes, scales = torch.ops.quaver.block_quantize(q, q_params.zero_point.to(torch.int64), to_scale.multiplier, to_scale.shift, to_value.multiplier, to_value.shift, number_format.name, scale_format.name, granularity.kind, granularity.axis,
                                                  granularity.block_size)
    scale         = scales.to(torch.float32) * scale_format.step_min * tensor_scale
    params        = QuantParams(scale, torch.zeros([1] * scale.ndim, device=scale.device), number_format, BlockScale(scales, scale_format, tensor_scale))
    return params_note(codes, params), params


@device_define("block_quantize")
def block_quantize_device(q: torch.Tensor, zero_point: torch.Tensor, scale_multiplier: torch.Tensor, scale_shift: torch.Tensor, value_multiplier: torch.Tensor, value_shift: torch.Tensor, out_format: str, scale_format: str, kind: GranularityKind,
                          axis: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    #* Device op "block_quantize": the code of each block scale from the largest |code| of its block, then each element divided by its block code, both rounded half to even.
    #? block_quantize_integer() calls it as torch.ops.quaver.block_quantize. Returns the element codes, and the block codes spread back so that they broadcast against q.
    #! The multipliers and shifts are static: scale_* turns one step of q into steps of the block scales at the top of out_format, value_* into steps of the elements.
    '''
    
    # The code of each block scale, from the largest |code| of the block (never below the smallest step)
    granularity = Granularity(kind, axis, block_size)
    centered    = q - zero_point
    largest     = group_split(centered.abs(), granularity).amax(dim=1)
    scales      = wide_round_format(largest, scale_multiplier, torch.ones_like(largest), scale_shift, build_format(scale_format), "half_even").clamp(min=1)
    scales      = group_merge(scales, q.shape, granularity)
    
    # Each element, divided by the code of its block scale, rounded half to even onto the format
    return wide_round_format(centered, value_multiplier, scales, value_shift, build_format(out_format), "half_even"), scales


def block_quantize_float(x: torch.Tensor, x_params: QuantParams, number_format: NumberFormat, scale_format: NumberFormat, tensor_scale: torch.Tensor, granularity: Granularity, style_name: str) -> tuple[torch.Tensor, QuantParams]:
    '''
    #* Float mode (calibration): return x as it is, with the block scales that its float values would get.
    '''
    
    block = compute_block(group_split(x.detach(), granularity).abs().amax(dim=1), tensor_scale, number_format, scale_format)
    codes = group_merge(format_to_integer(block, scale_format), x.shape, granularity)
    scale = group_merge(block * tensor_scale                  , x.shape, granularity)
    return x, QuantParams(scale, torch.zeros_like(scale), number_format, BlockScale(codes, scale_format, tensor_scale))
