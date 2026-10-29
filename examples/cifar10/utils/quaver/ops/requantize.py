# @file requantize.py
# @brief Rescale accumulators to output codes with a multiplier and a shift
# @author Kohei Yamaguchi, Claude

##########################################################
'''
Add a requantize style
1. Make RequantizeStyle(name, multiplier_max, shift_max, round_mode) for the device.
2. Add "name: style" to REQUANTIZE_REGISTRY.
3. Pass the name as style_name.
'''
##########################################################

import torch

from ..formats import RoundMode, build_format, device_define, round_half_away
from ..scales  import QuantParams, params_attach, params_codes, params_math_dtype, params_note, params_round
from .types    import FixedPoint, RequantizeStyle
from .wide     import wide_round_format

REQUANTIZE_REGISTRY: dict[str, RequantizeStyle] = {
    "mul16_shift": RequantizeStyle("mul16_shift", multiplier_max=2**16 - 1, shift_max=94, round_mode="half_up"  ),
    "shift_only" : RequantizeStyle("shift_only" , multiplier_max=1        , shift_max=94, round_mode="half_up"  ),
    "exact"      : RequantizeStyle("exact"      , multiplier_max=0        , shift_max=0 , round_mode="half_away"),
}



def build_requantize(name: str) -> RequantizeStyle:
    '''
    #* Return the requantize style for the name.
    #? mul16_shift: a 16-bit multiplier and (acc x M0 + 2^(n-1)) >> n. shift_only: power-of-two scales (M0 = 1). exact: the real ratio.
    #! Add new styles to REQUANTIZE_REGISTRY.
    '''
    
    if name not in REQUANTIZE_REGISTRY:
        raise KeyError(f"Unknown requantize style {name!r}. Available: {', '.join(REQUANTIZE_REGISTRY)}")
    return REQUANTIZE_REGISTRY[name]


def requantize_params(ratio: torch.Tensor, style_name: str) -> FixedPoint:
    '''
    #* Find the multiplier M0 and the shift n that stand for the ratio (ratio = M0 / 2^n), with one formula (no loop over the shifts).
    #? n is the largest shift where M0 = round(ratio x 2^n) still fits in multiplier_max, so M0 uses every bit (16 bits for mul16_shift). n < 0 is a left shift (ratios of 2^16 or more), down to -30.
    #! Only for static scales: the device gets M0 and n from Python. Ratios that change with the data use requantize_ratio(), which needs only integers.
    #! The op torch.ops.quaver.fixed_point finds them, so an exported model folds them into constants (or keeps one node when a scale is measured every time).
    '''
    
    # The exact style keeps the ratio itself
    style = build_requantize(style_name)
    ratio = ratio.detach().double()
    if style.multiplier_max == 0:
        return FixedPoint(ratio, torch.zeros_like(ratio, dtype=torch.int64), style)
    
    # A multiplier and a shift from the device op
    multiplier, shift = torch.ops.quaver.fixed_point(ratio, style_name)
    return FixedPoint(multiplier, shift, style)


@device_define("fixed_point")
def requantize_fixed_device(ratio: torch.Tensor, style_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    #* The op "fixed_point": the multiplier M0 and the shift n of a ratio (ratio = M0 / 2^n), with the largest n that keeps M0 at or below multiplier_max.
    #? requantize_params() calls it as torch.ops.quaver.fixed_point, for styles with a multiplier. ratio is float64.
    '''
    
    style       = build_requantize(style_name)
    _, exponent = torch.frexp(ratio)
    shift       = style.multiplier_max.bit_length() - exponent.to(torch.int64)
    shift       = torch.where(_requantize_multiplier(ratio, shift) > style.multiplier_max, shift - 1, shift).clamp(-30, style.shift_max)
    multiplier  = _requantize_multiplier(ratio, shift).clamp(0, style.multiplier_max)
    return multiplier.to(torch.int64), shift


def requantize_ratio(numerator: torch.Tensor, denominator: torch.Tensor, style_name: str) -> FixedPoint:
    '''
    #* Find M0 and n for numerator / denominator with integers only (shifts, a multiplication, and an integer division), as the device does it while running.
    #? NEQ and the attention scores use this for ratios that change with the data. M0 = round(numerator x 2^n / denominator), a half up, with the largest n that fits (n < 0 is a left shift).
    #! Pass int64 tensors with 1 <= numerator < 2^62 and 1 <= denominator < 2^46, so every step stays below 2^63. Styles with a multiplier only (not exact).
    #! frexp counts one bit too many for a numerator that float64 rounds up to a power of two. Then M0 starts one shift lower and needs one step less, so M0 and n stay the same.
    '''
    
    # Check here at the entry: the exact style keeps a real ratio, which the device cannot find while running
    style = build_requantize(style_name)
    if style.multiplier_max == 0:
        raise ValueError(f"The style {style_name!r} keeps a real ratio, but this ratio changes with the data and needs M0 and n. Use a style with a multiplier (mul16_shift or shift_only).")
    
    # The shift that puts M0 just under 2^bits, from the bit lengths (the true ratio can be up to 2 times off, so M0 is at most 2^(bits + 1))
    lengths    = [torch.frexp(values.double())[1].to(torch.int64) for values in (numerator, denominator)]
    shift      = style.multiplier_max.bit_length() - (lengths[0] - lengths[1])
    multiplier = _requantize_divide(numerator, denominator, shift)
    
    # Step the shift down while M0 is too large: two steps are always enough, since each one halves M0
    for _ in range(2):
        shift      = torch.where(multiplier > style.multiplier_max, shift - 1, shift)
        multiplier = _requantize_divide(numerator, denominator, shift)
    return FixedPoint(multiplier, shift, style)


def requantize_rescale(acc: torch.Tensor, fixed_point: FixedPoint, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    '''
    #* Multiply the accumulator by the ratio: acc x M0 / 2^n, before rounding, in the float type dtype.
    #? In float64 the product of a 32-bit accumulator and a 16-bit multiplier stays below 2^53, so the result is exact.
    #! In float32 the product keeps only 24 bits, so a value very near the middle of two codes can round to the other one.
    '''
    
    if fixed_point.style.multiplier_max == 0:
        return acc.to(dtype) * fixed_point.multiplier.to(dtype)
    return torch.ldexp(acc.to(dtype) * fixed_point.multiplier.to(dtype), -fixed_point.shift)


def requantize_fake(x: torch.Tensor, x_params: QuantParams, out_params: QuantParams, style_name: str) -> torch.Tensor:
    '''
    #* Training mode: requantize in the same way as the device, with a straight-through gradient (and the LSQ gradient for a learned output scale).
    #? It gets the codes back from x, and runs the integer math of requantize_integer in the type of params_math_dtype() (float64 unless precision_set_math() changed it).
    #! x must hold exact values of the codes (an accumulator, or the output of another op), so turn off TF32.
    '''
    
    dtype         = params_math_dtype()
    codes, inside = _requantize_codes(params_codes(x, x_params, dtype), x_params, out_params, style_name, dtype)
    return params_attach(x, codes, out_params, inside, dtype)


def requantize_integer(acc: torch.Tensor, acc_params: QuantParams, out_params: QuantParams, style_name: str) -> torch.Tensor:
    '''
    #* Integer mode: rescale the accumulator (or any codes) to output codes, like the device.
    #? The ratio is acc_params.code_scale / out_params.code_scale, as a multiplier and a shift. The product is rounded with int64 math only (wide_round_format), exact for any accumulator.
    #! The zero point of acc_params is removed first, so codes of an asymmetric format keep their real values. Accumulators have the zero point 0. The exact style uses float64.
    '''
    
    # The exact style keeps a real ratio, so it runs the float64 math of the training mode
    fixed_point = requantize_params(acc_params.code_scale / out_params.code_scale, style_name)
    if fixed_point.style.multiplier_max == 0:
        codes, _ = _requantize_codes(acc, acc_params, out_params, style_name, torch.float64)
        return codes
    
    # A multiplier and a shift: the device op, with M0 and n from here
    codes = torch.ops.quaver.requantize(acc, acc_params.zero_point.to(torch.int64), fixed_point.multiplier, fixed_point.shift, out_params.zero_point.to(torch.int64), out_params.number_format.name, fixed_point.style.round_mode)
    return params_note(codes, out_params)


@device_define("requantize")
def requantize_device(acc: torch.Tensor, zero_point: torch.Tensor, multiplier: torch.Tensor, shift: torch.Tensor, out_zero_point: torch.Tensor, out_format: str, mode: RoundMode) -> torch.Tensor:
    '''
    #* Device op "requantize": (acc - zero_point) x M0 / 2^n, rounded once onto out_format, plus the output zero point, stopped at the ends of the range.
    #? requantize_integer() calls it as torch.ops.quaver.requantize. M0 and n are static (requantize_params() finds them, and the device gets them).
    #! int64 math only (wide_round_format), exact for any width of acc. Only styles with a multiplier come here.
    '''
    
    centered = acc - zero_point
    return wide_round_format(centered, multiplier, torch.ones_like(centered), shift, build_format(out_format), mode, out_zero_point)


def requantize_float(x: torch.Tensor, x_params: QuantParams, out_params: QuantParams, style_name: str) -> torch.Tensor:
    '''
    #* Float mode (calibration): return x as it is, without requantizing.
    #? The layer measures the output range before this (Quantizer.measure()), so the scalers see the float values, like observers in PyTorch.
    '''
    
    return x


def _requantize_codes(acc: torch.Tensor, acc_params: QuantParams, out_params: QuantParams, style_name: str, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    #* Remove the zero point of the input, rescale it to output codes, and say which codes were inside the range.
    #? requantize_fake() and requantize_integer() use this, so both modes agree (always in float64, and nearly always in float32).
    #! The multiplier and the shift always come from float64, and only the math on each value uses dtype.
    '''
    
    fixed_point = requantize_params(acc_params.code_scale / out_params.code_scale, style_name)
    centered    = acc - acc_params.zero_point.to(torch.int64)
    return params_round(requantize_rescale(centered, fixed_point, dtype), out_params, fixed_point.style.round_mode)


def _requantize_multiplier(ratio: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    '''
    #* Round ratio x 2^shift to the multiplier. A half goes up (the ratio is positive).
    #? requantize_params() uses this twice.
    '''
    
    return round_half_away(torch.ldexp(ratio, shift))


def _requantize_divide(numerator: torch.Tensor, denominator: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    '''
    #* Return round(numerator x 2^shift / denominator) with an integer division (a half goes up). A negative shift moves the denominator up instead.
    #? requantize_ratio() uses this three times.
    #! The remainder decides the rounding, so nothing is doubled past the sizes that requantize_ratio() allows.
    '''
    
    up    = torch.bitwise_left_shift(numerator  , shift.clamp_min(0)   )
    down  = torch.bitwise_left_shift(denominator, (-shift).clamp_min(0))
    whole = torch.div(up, down, rounding_mode="floor")
    return whole + (2 * (up - whole * down) >= down).to(torch.int64)
