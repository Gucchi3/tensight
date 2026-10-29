# @file integer.py
# @brief INT formats (int2 to int8, uint2 to uint8, int32)
# @author Kohei Yamaguchi, Claude

import torch

from .types import NumberFormat



def integer_format(bits: int, signed: bool) -> NumberFormat:
    '''
    #* Make an INT format: int2 to int8 when signed, and uint2 to uint8 when not.
    #? The bias and the accumulator use int32 inside.
    '''
    
    low  = -(2 ** (bits - 1)) if signed else 0
    high = 2 ** (bits - 1) - 1 if signed else 2**bits - 1
    name = f"{'int' if signed else 'uint'}{bits}"
    return NumberFormat(name=name, kind="int", bits=bits, signed=signed, exponent_bits=0, mantissa_bits=0, exponent_bias=0, nan_top=False, min_value=float(low), max_value=float(high), step_min=1.0)


def integer_encode(values: torch.Tensor, number_format: NumberFormat) -> torch.Tensor:
    '''
    #* Turn values on the INT grid into int64 codes.
    #! Pass values that are already on the grid, for example from format_quantize().
    '''
    
    return values.to(torch.int64)


def integer_decode(codes: torch.Tensor, number_format: NumberFormat) -> torch.Tensor:
    '''
    #* Turn INT codes back into float32 values.
    '''
    
    return codes.to(torch.float32)
