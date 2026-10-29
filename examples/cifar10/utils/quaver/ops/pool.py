# @file pool.py
# @brief Average pooling on integer codes
# @author Kohei Yamaguchi, Claude

from torch.nn import functional as F

import math
import torch

from ..formats import RoundMode, device_define, ste_attach
from ..scales  import QuantParams, params_codes, params_math_dtype, params_named, params_note, params_round, params_values



def pool_avg2d_fake(x: torch.Tensor, x_params: QuantParams, kernel_size: int | tuple[int, int], stride: int | tuple[int, int] | None, mode: RoundMode) -> torch.Tensor:
    '''
    #* Training mode average pooling: the same integer math as pool_avg2d_integer, with straight-through gradients.
    #? The math uses the type of params_math_dtype().
    '''
    
    dtype = params_math_dtype()
    codes = _pool_avg2d_codes(params_codes(x, x_params, dtype), x_params, kernel_size, stride, mode, dtype)
    return ste_attach(F.avg_pool2d(x, kernel_size, stride), params_values(codes, x_params, dtype))


def pool_avg2d_integer(q_x: torch.Tensor, x_params: QuantParams, kernel_size: int | tuple[int, int], stride: int | tuple[int, int] | None, mode: RoundMode) -> torch.Tensor:
    '''
    #* Integer mode average pooling: add the codes of each window, divide by the count, and round.
    #? The output keeps the scale and the zero point of the input. The device op torch.ops.quaver.pool_avg2d does it.
    '''
    
    window = list(kernel_size) if isinstance(kernel_size, tuple) else [kernel_size, kernel_size]
    step   = window if stride is None else list(stride) if isinstance(stride, tuple) else [stride, stride]
    return params_note(torch.ops.quaver.pool_avg2d(q_x, x_params.zero_point.to(torch.int64), x_params.number_format.name, window, step, mode), x_params)


@device_define("pool_avg2d")
def pool_avg2d_device(q_x: torch.Tensor, zero_point: torch.Tensor, number_format: str, kernel_size: list[int], stride: list[int], mode: RoundMode) -> torch.Tensor:
    '''
    #* Device op "pool_avg2d": add the codes of each window (zero point removed), divide by the count, round onto the format, and add the zero point back.
    #? pool_avg2d_integer() calls it as torch.ops.quaver.pool_avg2d. The math is float64: the sums are exact integers, and the division is rounded correctly.
    '''
    
    return _pool_avg2d_codes(q_x, params_named(zero_point, number_format), tuple(kernel_size), tuple(stride), mode, torch.float64)


def pool_avg2d_float(x: torch.Tensor, x_params: QuantParams, kernel_size: int | tuple[int, int], stride: int | tuple[int, int] | None, mode: RoundMode) -> torch.Tensor:
    '''
    #* Float mode (calibration): the float average pooling, without rounding.
    '''
    
    return F.avg_pool2d(x, kernel_size, stride)


def _pool_avg2d_codes(q_x: torch.Tensor, x_params: QuantParams, kernel_size: int | tuple[int, ...], stride: int | tuple[int, ...] | None, mode: RoundMode, dtype: torch.dtype) -> torch.Tensor:
    '''
    #* Add the centered codes of each window, divide by the count, and round back onto the format.
    #? pool_avg2d_fake() and pool_avg2d_device() use this, so both modes agree. Only the zero point and the format of x_params are used.
    #! The division of these integers is rounded correctly, so every rounding mode is exact (in float32 too, while the sums stay below 2^24).
    '''
    
    # Add the centered codes of each window
    centered = (q_x - x_params.zero_point.to(torch.int64)).to(dtype)
    total    = F.avg_pool2d(centered, kernel_size, stride, divisor_override=1)
    count    = math.prod(kernel_size) if isinstance(kernel_size, tuple) else kernel_size * kernel_size
    
    # Take the mean, and round it back onto the format
    codes, _ = params_round(total / count, x_params, mode)
    return codes
