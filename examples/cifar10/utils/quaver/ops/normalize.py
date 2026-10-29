# @file normalize.py
# @brief NEQ: L1 normalization and quantization in one step, for each token or for the tensor (SimA), and the requantization of their product
# @author Kohei Yamaguchi, Claude

import torch

from ..formats   import NumberFormat, build_format, device_define, ste_attach
from ..scales    import QuantParams, params_attach, params_codes, params_math_dtype, params_named, params_note, params_round, params_values
from .accumulate import ACCUMULATOR_FORMAT
from .requantize import build_requantize, requantize_params, requantize_ratio, requantize_rescale
from .types      import FixedPoint, NeqScope

NORM_MIN          = torch.finfo(torch.float32).eps
NORM_BITS         = 20
NORM_INVERSE_BITS = 24



def normalize_l1_fake(x: torch.Tensor, x_params: QuantParams, number_format: NumberFormat, style_name: str, scope: NeqScope = "tensor") -> tuple[torch.Tensor, QuantParams]:
    '''
    #* Training mode NEQ: x / L1(x) for each token (last axis), quantized to number_format with the division put into the scale.
    #? It runs the same integer math as normalize_l1_integer, in the type of params_math_dtype(). The gradients follow the float normalization.
    #! x is usually a conv accumulator (fake values), so turn off TF32 to get its codes back exactly.
    '''
    
    dtype       = params_math_dtype()
    align       = requantize_params(x_params.code_scale / x_params.scale.amax(), style_name)
    codes, peak = _normalize_l1_codes(params_codes(x, x_params, dtype), x_params.zero_point, align, number_format, style_name, scope, dtype)
    scale       = (peak.double() / (2**NORM_BITS * number_format.max_value)).float()
    params      = QuantParams(scale, torch.zeros([1] * peak.ndim, device=peak.device), number_format, peak=peak)
    reference   = x / x.abs().sum(dim=-1, keepdim=True).clamp_min(NORM_MIN)
    return ste_attach(reference, params_values(codes, params, dtype)), params


def normalize_l1_integer(q_x: torch.Tensor, x_params: QuantParams, number_format: NumberFormat, style_name: str, scope: NeqScope = "tensor") -> tuple[torch.Tensor, QuantParams]:
    '''
    #* Integer mode NEQ (Normalization-Equivalent Quantization): L1-normalize each token (last axis) and quantize it in one step, with 1 / L1 in the scale.
    #? scope "tensor" is Algorithm 2 of the paper (its deployed setting): one scale for each sample and head, from the token with the largest max|v| / L1. "token": one scale for each token.
    #? The device op torch.ops.quaver.normalize_l1 does it, with the static M0 and n from here that put the channels on the largest channel scale.
    #? The parameters keep the integer p (peak), which the attention map uses. Their scale is p / (2^NORM_BITS x largest value of the format).
    #! Integers only: p = floor(max|v| x 2^NORM_BITS / L1) (the largest over the tokens for "tensor"), M0 = round(top x 2^NORM_BITS x 2^n / (L1 x p)), and code = (v x M0 + 2^(n-1)) >> n.
    '''
    
    align       = requantize_params(x_params.code_scale / x_params.scale.amax(), style_name)
    codes, peak = torch.ops.quaver.normalize_l1(q_x, x_params.zero_point.to(torch.int64), align.multiplier, align.shift, number_format.name, style_name, scope)
    scale       = (peak.double() / (2**NORM_BITS * number_format.max_value)).float()
    params      = QuantParams(scale, torch.zeros([1] * peak.ndim, device=peak.device), number_format, peak=peak)
    return params_note(codes, params), params


@device_define("normalize_l1")
def normalize_l1_device(q_x: torch.Tensor, zero_point: torch.Tensor, multiplier: torch.Tensor, shift: torch.Tensor, out_format: str, style_name: str, scope: NeqScope) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    #* Device op "normalize_l1" (NEQ): put the channels on one scale (static M0 and n), measure each token with integers, and rescale and round each value with the integer multiplier of NEQ.
    #? normalize_l1_integer() calls it as torch.ops.quaver.normalize_l1. Returns the codes and the integer p (int64, 1 to 2^NORM_BITS), which sets their scale: p / (2^NORM_BITS x largest value of out_format).
    '''
    
    return _normalize_l1_codes(q_x, zero_point, FixedPoint(multiplier, shift, build_requantize(style_name)), build_format(out_format), style_name, scope, torch.float64)


def normalize_l1_float(x: torch.Tensor, x_params: QuantParams, number_format: NumberFormat, style_name: str, scope: NeqScope = "tensor") -> tuple[torch.Tensor, QuantParams]:
    '''
    #* Float mode (calibration): the float L1 normalization of each token (last axis), and the scale that NEQ would give it (for each token, or one for the tensor).
    #? The scale puts the largest |value| on the largest value of the format, like the other modes.
    '''
    
    normalized = x / x.abs().sum(dim=-1, keepdim=True).clamp_min(NORM_MIN)
    largest    = normalized.abs().amax(dim=-1 if scope == "token" else (-2, -1), keepdim=True)
    scale      = (largest / number_format.max_value).clamp_min(NORM_MIN)
    return normalized, QuantParams(scale, torch.zeros_like(scale), number_format)


def normalize_l1_scores_fake(x: torch.Tensor, x_params: QuantParams, q_params: QuantParams, k_params: QuantParams, out_params: QuantParams, factor: float, style_name: str) -> torch.Tensor:
    '''
    #* Training mode: requantize the product Q K^T of two NEQ outputs (the attention map) like normalize_l1_scores_integer, with straight-through gradients.
    #? x is the fake product from the matrix product (x_params), and the reference for the gradients is x x factor. A learned output scale gets the LSQ gradient.
    '''
    
    # Check here at the entry: Q and K must come from NEQ, which gives their integer p
    if q_params.peak is None or k_params.peak is None:
        raise ValueError("q_params and k_params must be the outputs of NEQ (normalize_l1), with their integer p (peak).")
    
    # The static part H, and the integer math of the device
    dtype         = params_math_dtype()
    tops          = [params.number_format.max_value / params.number_format.step_min for params in (q_params, k_params)]
    static        = requantize_params(factor / (tops[0] * tops[1] * out_params.code_scale), style_name)
    codes, inside = _normalize_l1_scores_codes(params_codes(x, x_params, dtype), q_params.peak, k_params.peak, static, out_params, dtype)
    return params_attach(x * factor, codes, out_params, inside, dtype)


def normalize_l1_scores_integer(acc: torch.Tensor, x_params: QuantParams, q_params: QuantParams, k_params: QuantParams, out_params: QuantParams, factor: float, style_name: str) -> torch.Tensor:
    '''
    #* Integer mode: requantize the exact product Q K^T of two NEQ outputs (the attention map) to the output codes, with a multiplier that the device finds with integers.
    #? The ratio is p_q x p_k x H: p_q and p_k are the integer maxima of NEQ, and H = factor / (top_q x top_k x output code scale) is static (M0 and n from Python).
    #? The device op torch.ops.quaver.normalize_l1_scores does it, with M0_H and n_H from here and the integer p of Q and K (peak of their parameters).
    #! The device cuts p_q x p_k x M0_H to a 16-bit M0 and a shift with integers. factor is 1 / sqrt(key_dim) in SimA. One multiplier for each sample and head, or for each pair.
    '''
    
    # Check here at the entry: Q and K must come from NEQ, which gives their integer p
    if q_params.peak is None or k_params.peak is None:
        raise ValueError("q_params and k_params must be the outputs of NEQ (normalize_l1), with their integer p (peak).")
    
    # The static part H, and the device op
    tops   = [params.number_format.max_value / params.number_format.step_min for params in (q_params, k_params)]
    static = requantize_params(factor / (tops[0] * tops[1] * out_params.code_scale), style_name)
    codes  = torch.ops.quaver.normalize_l1_scores(acc, q_params.peak, k_params.peak, static.multiplier, static.shift, out_params.zero_point.to(torch.int64), out_params.number_format.name, style_name)
    return params_note(codes, out_params)


@device_define("normalize_l1_scores")
def normalize_l1_scores_device(acc: torch.Tensor, q_peak: torch.Tensor, k_peak: torch.Tensor, multiplier: torch.Tensor, shift: torch.Tensor, out_zero_point: torch.Tensor, out_format: str, style_name: str) -> torch.Tensor:
    '''
    #* Device op "normalize_l1_scores": requantize Q K^T of two NEQ outputs, with p_q x p_k x M0_H cut to M0 and n with integers.
    #? normalize_l1_scores_integer() calls it as torch.ops.quaver.normalize_l1_scores. M0_H and n_H (multiplier and shift) are static. q_peak and k_peak are the integers p of NEQ (from normalize_l1_device).
    '''
    
    static   = FixedPoint(multiplier, shift, build_requantize(style_name))
    codes, _ = _normalize_l1_scores_codes(acc, q_peak, k_peak, static, params_named(out_zero_point, out_format), torch.float64)
    return codes


def normalize_l1_scores_float(x: torch.Tensor, x_params: QuantParams, q_params: QuantParams, k_params: QuantParams, out_params: QuantParams, factor: float, style_name: str) -> torch.Tensor:
    '''
    #* Float mode (calibration): the float product times factor, without requantizing.
    '''
    
    return x * factor


def _normalize_l1_codes(q_x: torch.Tensor, zero_point: torch.Tensor, align: FixedPoint, number_format: NumberFormat, style_name: str, scope: NeqScope, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    #* Put the channels on one scale, measure each token with integers, and rescale and round each value with the integer multiplier of NEQ. Returns the codes and the integer p.
    #? normalize_l1_fake() and normalize_l1_device() use this, so both modes agree. align puts each channel on the largest channel scale (static). Every other multiplier comes from integer divisions. Only the rescaling of the values uses dtype.
    #! The scale of the codes is p / (2^NORM_BITS x largest value of the format), with an integer p from 1 to 2^NORM_BITS.
    '''
    
    # Put every channel on the largest channel scale, like a requantization to INT32 (static M0 and n)
    style      = build_requantize(style_name)
    aligned, _ = params_round(requantize_rescale(q_x - zero_point.to(torch.int64), align, dtype), params_named(torch.zeros((), dtype=torch.int64, device=q_x.device), ACCUMULATOR_FORMAT.name), style.round_mode)
    
    # Measure each token: the L1 norm, and the normalized maximum p = floor(max|v| x 2^NORM_BITS / L1), the largest over the tokens for one scale for the tensor
    total = aligned.abs().sum(dim=-1, keepdim=True).clamp_min(1)
    peak  = torch.div(torch.bitwise_left_shift(aligned.abs().amax(dim=-1, keepdim=True), NORM_BITS), total, rounding_mode="floor")
    peak  = (peak if scope == "token" else peak.amax(dim=-2, keepdim=True)).clamp_min(1)
    
    # The multiplier of each token, top x 2^NORM_BITS / (L1 x p), from two integer divisions: inverse = round(top x 2^(NORM_BITS + NORM_INVERSE_BITS) / p), then inverse / L1 as M0 and n
    top        = torch.full_like(peak, round(number_format.max_value / number_format.step_min) << (NORM_BITS + NORM_INVERSE_BITS))
    whole      = torch.div(top, peak, rounding_mode="floor")
    inverse    = whole + (2 * (top - whole * peak) >= peak).to(torch.int64)
    multiplier = requantize_ratio(inverse, total, style_name)
    fixed      = FixedPoint(multiplier.multiplier, multiplier.shift + NORM_INVERSE_BITS, style)
    
    # Rescale and round each value onto the format
    codes, _ = params_round(requantize_rescale(aligned, fixed, dtype), params_named(torch.zeros_like(peak), number_format.name), style.round_mode)
    return codes, peak


def _normalize_l1_scores_codes(acc: torch.Tensor, q_peak: torch.Tensor, k_peak: torch.Tensor, static: FixedPoint, out_params: QuantParams, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    #* Requantize Q K^T of two NEQ outputs with the integer maxima p of NEQ, and say which codes were inside the range.
    #? normalize_l1_scores_fake() and normalize_l1_scores_device() use this, so both modes agree. static is H = factor / (top_q x top_k x output code scale) as M0 and n (Python gives these to the device).
    #! The product acc x M0 stays exact in float64 while |acc| < 2^37, which holds for codes up to 8 bits and up to 2^20 values in each sum. Only the zero point and the format of out_params are used.
    '''
    
    # The device cuts p_q x p_k x M0_H to M0 and n with integers, and the two scales of 2^NORM_BITS go into the shift
    style = static.style
    pair  = requantize_ratio(q_peak * k_peak * static.multiplier, torch.ones_like(static.multiplier), style.name)
    fixed = FixedPoint(pair.multiplier, pair.shift + static.shift + 2 * NORM_BITS, style)
    return params_round(requantize_rescale(acc, fixed, dtype), out_params, style.round_mode)
