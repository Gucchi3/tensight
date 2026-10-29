# @file compute.py
# @brief Turn ranges of values into scales and zero points
# @author Kohei Yamaguchi, Claude

import torch

from ..formats import NumberFormat, format_quantize, round_half_even

SCALE_MIN = torch.finfo(torch.float32).tiny
MSE_STEPS = 100



def compute_scale(low: torch.Tensor, high: torch.Tensor, number_format: NumberFormat, symmetric: bool) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    #* Turn a range of values into a scale and a zero point.
    #? Symmetric works for every format. Asymmetric is for INT formats.
    #! The asymmetric range is made wider to include 0, so 0 stays exact.
    '''
    
    # Symmetric: the larger of |low| and |high| goes to the largest value of the format, and the zero point is 0
    if symmetric:
        scale = (torch.maximum(low.abs(), high.abs()) / number_format.max_value).clamp_min(SCALE_MIN)
        return scale, torch.zeros_like(scale)
    
    # Asymmetric: make the range include 0, spread it over every code, and put real 0 on an integer code
    low        = low.clamp(max=0.0)
    high       = high.clamp(min=0.0)
    scale      = ((high - low) / (number_format.max_value - number_format.min_value)).clamp_min(SCALE_MIN)
    zero_point = round_half_even(number_format.min_value - low / scale).clamp(number_format.min_value, number_format.max_value)
    return scale, zero_point


def compute_mse_range(values: torch.Tensor, low: torch.Tensor, high: torch.Tensor, number_format: NumberFormat, symmetric: bool) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    #* Shrink the range of each group to the one whose quantized values have the least squared error (the MSE start values of LSQ+).
    #? values has one row for each group, and low and high are the measured ranges. Symmetric ranges shrink both ends together, asymmetric ones shrink the low end and the high end in turn (twice).
    #! It tries MSE_STEPS fractions of the measured ends and rounds half to even. It makes no GPU sync, so it can run inside the first training step.
    '''
    
    # The fractions to try, the ends that move in each turn, and the range that is really used (asymmetric ranges include 0)
    fractions = torch.arange(1, MSE_STEPS + 1, dtype=values.dtype, device=values.device) / MSE_STEPS
    turns     = ["both"] if symmetric else ["low", "high", "low", "high"]
    low, high = (low, high) if symmetric else (low.clamp(max=0.0), high.clamp(min=0.0))
    best      = (low, high)
    for turn in turns:
        # Try each fraction of the moving end, and keep the range with the least error in each group
        start = best
        error = torch.full_like(low, torch.inf)
        for fraction in fractions:
            trial       = (start[0] if turn == "high" else low * fraction, start[1] if turn == "low" else high * fraction)
            scale, zero = compute_scale(trial[0], trial[1], number_format, symmetric)
            quantized   = (format_quantize(values / scale[:, None], number_format, "half_even", zero[:, None]) - zero[:, None]) * scale[:, None]
            trial_error = (quantized - values).square().mean(dim=1)
            better      = trial_error < error
            error       = torch.where(better, trial_error, error)
            best        = (torch.where(better, trial[0], best[0]), torch.where(better, trial[1], best[1]))
    return best


def compute_pow2(scale: torch.Tensor) -> torch.Tensor:
    '''
    #* Round the scale up to a power of two, for devices that divide only with shifts.
    #! This is exact, because frexp and ldexp only change the exponent.
    '''
    
    mantissa, exponent = torch.frexp(scale)
    return torch.ldexp(torch.ones_like(scale), exponent - (mantissa == 0.5).to(exponent.dtype))


def compute_block(block_amax: torch.Tensor, tensor_scale: torch.Tensor, number_format: NumberFormat, scale_format: NumberFormat) -> torch.Tensor:
    '''
    #* Return the scale of each block of a block-scaled format: block_amax / top / tensor scale, rounded to scale_format (half to even).
    #? top is the largest value of number_format (6 for fp4, 7 for ufp4). scale_format ue4m3 gives NVFP4, and an unsigned INT format (uint6, uint10) gives integer block scales (VS-Quant).
    #! Block scales stay at or above the smallest step of scale_format, so a block of zeros still has a scale.
    '''
    
    return format_quantize(block_amax / number_format.max_value / tensor_scale, scale_format, "half_even").clamp_min(scale_format.step_min)
