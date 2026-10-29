# @file accumulate.py
# @brief Exact accumulators of conv, linear, and matmul (INT32, or wider for block-scaled tensors)
# @author Kohei Yamaguchi, Claude

from collections.abc import Callable
from torch.nn        import functional as F

import torch

from ..formats import NumberFormat, device_define, integer_format
from ..scales  import QuantParams, params_bits, params_integers, params_named, params_unit

# INT32 accumulators. Overflow is not modeled: the sums are exact (int64), and the device is assumed to stay inside INT32
ACCUMULATOR_FORMAT = integer_format(32, signed=True)

# Accumulators of block-scaled tensors: every block is put on one step before the sum, so they need more than 32 bits (54 for NVFP4 with K = 1024)
WIDE_ACCUMULATOR_FORMAT = integer_format(62, signed=True)

# One float64 conv or matmul is exact while its sums stay below 2^EXACT_BITS (the error of any algorithm stays far below 0.5)
EXACT_BITS = 46



def accumulate_scale(x_params: QuantParams, w_params: QuantParams) -> torch.Tensor:
    '''
    #* Return the scale of the accumulator of a conv or linear layer, one for each output channel (flat).
    #? Accumulator scale = step of the input integers x step of the weight integers (params_unit: the code scale, or for block-scaled tensors the one step of all their blocks).
    #! The input must have one step for the tensor. The weight must have one for the tensor or one for each output channel (axis 0).
    '''
    
    return params_unit(x_params).reshape(-1) * params_unit(w_params).reshape(-1)


def accumulate_format(x_params: QuantParams, w_params: QuantParams) -> NumberFormat:
    '''
    #* Return the format of the accumulator: INT32, or the wide format when a block-scaled tensor is in the sum.
    #! The math keeps this. model_export() writes int32 instead of the wide format when accumulate_bits() is 32 or less.
    '''
    
    return WIDE_ACCUMULATOR_FORMAT if x_params.block is not None or w_params.block is not None else ACCUMULATOR_FORMAT


def accumulate_bits(x_params: QuantParams, w_params: QuantParams, count: int, bias_peak: int = 0) -> int:
    '''
    #* Return the bits (with the sign) that a sum of count products and a bias can need in the worst case, from the formats and the largest |bias code|.
    #? Without a bias: params_bits() of both sides + ceil(log2(count)) + 1. A bias adds one more bit only when it can carry the sum past that. model_export() uses this.
    '''
    
    largest = 2 ** (params_bits(x_params) + params_bits(w_params) + (count - 1).bit_length()) - 1
    return (largest + bias_peak).bit_length() + 1


def accumulate_conv2d(q_x: torch.Tensor, x_params: QuantParams, q_w: torch.Tensor, w_params: QuantParams, stride: int | tuple[int, int] = 1, padding: int | tuple[int, int] = 0, dilation: int | tuple[int, int] = 1, groups: int = 1) -> torch.Tensor:
    '''
    #* Return the exact accumulator (int64) of a conv on integer codes.
    #? The zero points are removed first, so the padding adds exact zeros. Block-scaled codes are multiplied by the codes of their block scales first. The device op torch.ops.quaver.accumulate_conv2d does it.
    '''
    
    x, w  = _accumulate_parts(x_params), _accumulate_parts(w_params)
    pairs = [list(value) if isinstance(value, tuple) else [value, value] for value in (stride, padding, dilation)]
    return torch.ops.quaver.accumulate_conv2d(q_x, x[0], x[1], q_w, w[0], w[1], x[2], x[3], w[2], w[3], accumulate_format(x_params, w_params).name, pairs[0], pairs[1], pairs[2], groups)


def accumulate_linear(q_x: torch.Tensor, x_params: QuantParams, q_w: torch.Tensor, w_params: QuantParams) -> torch.Tensor:
    '''
    #* Return the exact accumulator (int64) of a linear layer on integer codes.
    #? The device op torch.ops.quaver.accumulate_linear does it.
    '''
    
    x, w = _accumulate_parts(x_params), _accumulate_parts(w_params)
    return torch.ops.quaver.accumulate_linear(q_x, x[0], x[1], q_w, w[0], w[1], x[2], x[3], w[2], w[3], accumulate_format(x_params, w_params).name)


def accumulate_matmul(q_a: torch.Tensor, a_params: QuantParams, q_b: torch.Tensor, b_params: QuantParams) -> torch.Tensor:
    '''
    #* Return the exact accumulator (int64) of a matrix product of integer codes.
    #? The device op torch.ops.quaver.accumulate_matmul does it.
    '''
    
    a, b = _accumulate_parts(a_params), _accumulate_parts(b_params)
    return torch.ops.quaver.accumulate_matmul(q_a, a[0], a[1], q_b, b[0], b[1], a[2], a[3], b[2], b[3], accumulate_format(a_params, b_params).name)


@device_define("accumulate_conv2d")
def accumulate_conv2d_device(q_x: torch.Tensor, x_zero_point: torch.Tensor, x_block: torch.Tensor | None, q_w: torch.Tensor, w_zero_point: torch.Tensor, w_block: torch.Tensor | None, x_format: str, x_scale_format: str, w_format: str,
                             w_scale_format: str, out_format: str, stride: list[int], padding: list[int], dilation: list[int], groups: int) -> torch.Tensor:
    '''
    #* Device op "accumulate_conv2d": the exact sum of (code - zero point) x (code of the block scale) products of a conv, as int64.
    #? x_block and w_block are the codes of the block scales of each value (None for tensors without blocks), and the scale formats name them ("" without blocks).
    #! out_format names the accumulator for the device: int32, or int62 when a block-scaled tensor is in the sum. The sum itself is exact (int64).
    '''
    
    x_params = params_named(x_zero_point, x_format, x_block, x_scale_format)
    w_params = params_named(w_zero_point, w_format, w_block, w_scale_format)
    return _accumulate_exact(lambda x, w: F.conv2d(x, w, None, stride, padding, dilation, groups), q_x, x_params, q_w, w_params, q_w[0].numel())


@device_define("accumulate_linear")
def accumulate_linear_device(q_x: torch.Tensor, x_zero_point: torch.Tensor, x_block: torch.Tensor | None, q_w: torch.Tensor, w_zero_point: torch.Tensor, w_block: torch.Tensor | None, x_format: str, x_scale_format: str, w_format: str,
                             w_scale_format: str, out_format: str) -> torch.Tensor:
    '''
    #* Device op "accumulate_linear": the exact sum of the products of a linear layer, as int64, like accumulate_conv2d_device().
    '''
    
    x_params = params_named(x_zero_point, x_format, x_block, x_scale_format)
    w_params = params_named(w_zero_point, w_format, w_block, w_scale_format)
    return _accumulate_exact(F.linear, q_x, x_params, q_w, w_params, q_w.shape[-1])


@device_define("accumulate_matmul")
def accumulate_matmul_device(q_a: torch.Tensor, a_zero_point: torch.Tensor, a_block: torch.Tensor | None, q_b: torch.Tensor, b_zero_point: torch.Tensor, b_block: torch.Tensor | None, a_format: str, a_scale_format: str, b_format: str,
                             b_scale_format: str, out_format: str) -> torch.Tensor:
    '''
    #* Device op "accumulate_matmul": the exact sum of the products of a matrix product a @ b, as int64, like accumulate_conv2d_device().
    '''
    
    a_params = params_named(a_zero_point, a_format, a_block, a_scale_format)
    b_params = params_named(b_zero_point, b_format, b_block, b_scale_format)
    return _accumulate_exact(accumulate_product, q_a, a_params, q_b, b_params, q_a.shape[-1])


def accumulate_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    '''
    #* Return a @ b. When one of the sizes is 1, use a broadcast product and a sum instead.
    #! Some CUDA builds of PyTorch (seen with 2.13) send products with a size of 1 to a Triton kernel that compiles C the first time.
    '''
    
    if 1 in (a.shape[-2], a.shape[-1], b.shape[-1]):
        return (a.unsqueeze(-1) * b.unsqueeze(-3)).sum(dim=-2)
    return torch.matmul(a, b)


def _accumulate_parts(params: QuantParams) -> tuple[torch.Tensor, torch.Tensor | None, str, str]:
    '''
    #* Return what the device op needs from the parameters: the zero point (int64), the codes of the block scales (or None), the format name, and the scale format name ("" without blocks).
    #? accumulate_conv2d(), accumulate_linear(), and accumulate_matmul() use this for both sides.
    '''
    
    if params.block is None:
        return params.zero_point.to(torch.int64), None, params.number_format.name, ""
    return params.zero_point.to(torch.int64), params.block.codes, params.number_format.name, params.block.scale_format.name


def _accumulate_exact(op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], q_a: torch.Tensor, a_params: QuantParams, q_b: torch.Tensor, b_params: QuantParams, size: int) -> torch.Tensor:
    '''
    #* Run op (a conv, linear, or matmul that sums size products) on the integers of both sides, exactly, as int64.
    #? The device ops use this. When one float64 op could pass 2^EXACT_BITS, b is cut into parts of fewer bits, and the parts are added at their places.
    #! The widths come from the formats (params_bits), so nothing is read from the GPU. A sum that needs more than 62 bits raises ValueError.
    '''
    
    # The integers of both sides, and the widths of the sum and of one part of b
    a, b       = params_integers(q_a, a_params), params_integers(q_b, b_params)
    bits_a     = params_bits(a_params)
    bits_b     = params_bits(b_params)
    bits_sum   = size.bit_length()
    part       = EXACT_BITS - bits_a - bits_sum
    if bits_a + bits_b + bits_sum > 62 or part < 1:
        raise ValueError(f"The exact sum needs {bits_a} + {bits_b} + {bits_sum} bits, more than int64 and float64 parts keep. Use narrower formats or fewer values in one sum.")
    
    # Each part of b (the lower parts are 0 to 2^part - 1, and the top part keeps the sign), one exact float64 op each, added at its place
    count = -(-bits_b // part)
    total = torch.round(op(a.double(), (b if count == 1 else b & (2**part - 1)).double())).to(torch.int64)
    for index in range(1, count):
        digit = b >> (index * part) if index == count - 1 else (b >> (index * part)) & (2**part - 1)
        total = total + (torch.round(op(a.double(), digit.double())).to(torch.int64) << (index * part))
    return total
