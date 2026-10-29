# @file add.py
# @brief Residual add of two quantized tensors
# @author Kohei Yamaguchi, Claude

import torch

from ..formats   import device_define
from ..scales    import QuantParams, params_attach, params_codes, params_math_dtype, params_named, params_note, params_round
from .requantize import build_requantize, requantize_params, requantize_rescale
from .types      import FixedPoint



def add_fake(a: torch.Tensor, a_params: QuantParams, b: torch.Tensor, b_params: QuantParams, out_params: QuantParams, style_name: str) -> torch.Tensor:
    '''
    #* Training mode residual add: the same integer math as add_integer, with straight-through gradients.
    #? A learned output scale (LSQ) also gets its gradient. The math uses the type of params_math_dtype().
    '''
    
    dtype         = params_math_dtype()
    fixed         = [requantize_params(params.code_scale / out_params.code_scale, style_name) for params in (a_params, b_params)]
    codes, inside = _add_codes(params_codes(a, a_params, dtype), a_params.zero_point, fixed[0], params_codes(b, b_params, dtype), b_params.zero_point, fixed[1], out_params, dtype)
    return params_attach(a + b, codes, out_params, inside, dtype)


def add_integer(q_a: torch.Tensor, a_params: QuantParams, q_b: torch.Tensor, b_params: QuantParams, out_params: QuantParams, style_name: str) -> torch.Tensor:
    '''
    #* Integer mode residual add: rescale both inputs to the output code units without rounding, add them, and round the sum once.
    #? On the device: (code - zero point) x M0 of each input is shifted to the larger shift n, the two are added, and the sum is rounded once with n. TFLite and CMSIS-NN also round only after the sum.
    #? The device op torch.ops.quaver.add does this, with the static M0 and n of each input from here.
    #! The math is always float64. The sum is exact when |sum| x 2^n < 2^53: for sums inside the output range this holds up to n = 40 (INT8), and larger sums are clipped anyway.
    '''
    
    fixed = [requantize_params(params.code_scale / out_params.code_scale, style_name) for params in (a_params, b_params)]
    codes = torch.ops.quaver.add(q_a, a_params.zero_point.to(torch.int64), fixed[0].multiplier, fixed[0].shift, q_b, b_params.zero_point.to(torch.int64), fixed[1].multiplier, fixed[1].shift, out_params.zero_point.to(torch.int64),
                               out_params.number_format.name, style_name)
    return params_note(codes, out_params)


@device_define("add")
def add_device(q_a: torch.Tensor, a_zero_point: torch.Tensor, a_multiplier: torch.Tensor, a_shift: torch.Tensor, q_b: torch.Tensor, b_zero_point: torch.Tensor, b_multiplier: torch.Tensor, b_shift: torch.Tensor, out_zero_point: torch.Tensor,
               out_format: str, style_name: str) -> torch.Tensor:
    '''
    #* Device op "add": (code - zero point) x M0 / 2^n of each input, added without rounding, then rounded once onto out_format, plus the output zero point.
    #? add_integer() calls it as torch.ops.quaver.add. The rounding is the one of the style (FP outputs round half to even).
    '''
    
    style    = build_requantize(style_name)
    codes, _ = _add_codes(q_a, a_zero_point, FixedPoint(a_multiplier, a_shift, style), q_b, b_zero_point, FixedPoint(b_multiplier, b_shift, style), params_named(out_zero_point, out_format), torch.float64)
    return codes


def add_float(a: torch.Tensor, a_params: QuantParams, b: torch.Tensor, b_params: QuantParams, out_params: QuantParams, style_name: str) -> torch.Tensor:
    '''
    #* Float mode (calibration): the float sum a + b, without requantizing.
    '''
    
    return a + b


def _add_codes(q_a: torch.Tensor, a_zero: torch.Tensor, a_fixed: FixedPoint, q_b: torch.Tensor, b_zero: torch.Tensor, b_fixed: FixedPoint, out_params: QuantParams, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    #* Rescale each input exactly to the output code units, add the two parts, round the sum once onto the output format, and say which codes were inside the range.
    #? add_fake() and add_device() use this, so both modes agree (always in float64, and nearly always in float32).
    #! Only the zero point and the format of out_params are used.
    '''
    
    # Rescale each input (zero point removed) to the output code units, without rounding
    parts = [requantize_rescale(codes - zero.to(torch.int64), fixed, dtype) for codes, zero, fixed in ((q_a, a_zero, a_fixed), (q_b, b_zero, b_fixed))]
    
    # Add the exact parts, and round the sum once onto the output format (with the rounding of the style, or half even for FP outputs)
    return params_round(parts[0] + parts[1], out_params, a_fixed.style.round_mode)
