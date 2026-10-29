# @file fp.py
# @brief Small FP formats (FP4, UFP4, and UE4M3 for block scales)
# @author Kohei Yamaguchi, Claude

import torch

from .types import NumberFormat



def fp_format(exponent_bits: int, mantissa_bits: int, signed: bool, nan_top: bool, name: str = "") -> NumberFormat:
    '''
    #* Make a small FP format from its exponent bits and mantissa bits.
    #? Unsigned formats (UFP) use the sign bit as one more mantissa bit: UFP4 is E2M2. name="" names the format by its kind and bits (fp4, ufp4).
    #! nan_top=True keeps the top code for NaN (like E4M3), so the largest value is one step lower.
    '''
    
    # Find the range of the exponent
    bias     = 2 ** (exponent_bits - 1) - 1
    low_exp  = 1 - bias
    high_exp = 2**exponent_bits - 1 - bias
    
    # Find the largest value, the number of bits, and the name
    top_mantissa = 2**mantissa_bits - (2 if nan_top else 1)
    max_value    = 2.0**high_exp * (1 + top_mantissa / 2**mantissa_bits)
    bits         = exponent_bits + mantissa_bits + (1 if signed else 0)
    name         = name or f"{'fp' if signed else 'ufp'}{bits}"
    return NumberFormat(name=name, kind="fp", bits=bits, signed=signed, exponent_bits=exponent_bits, mantissa_bits=mantissa_bits, exponent_bias=bias, nan_top=nan_top, min_value=-max_value if signed else 0.0, max_value=max_value,
                        step_min=2.0 ** (low_exp - mantissa_bits))


def fp_step(x: torch.Tensor, number_format: NumberFormat) -> torch.Tensor:
    '''
    #* Return the grid step of the FP format at each value of x.
    #? Values in [2^e, 2^(e+1)) have the step 2^(e - mantissa_bits). Very small values (subnormals) use the step of the lowest exponent.
    #! This is exact, because frexp and ldexp only change the exponent.
    '''
    
    exponent = _fp_exponent(x, number_format)
    return torch.ldexp(torch.ones_like(x), exponent - number_format.mantissa_bits)


def fp_encode(values: torch.Tensor, number_format: NumberFormat) -> torch.Tensor:
    '''
    #* Turn values on the FP grid into bit codes (sign, exponent, mantissa) as int64.
    #! Pass values that are already on the grid, for example from format_quantize().
    '''
    
    # Find the exponent of each absolute value
    size      = values.abs()
    exponent  = _fp_exponent(size, number_format)
    subnormal = size < 2.0 ** (1 - number_format.exponent_bias)
    
    # The exponent field is 0 for subnormals, and the mantissa is the position between two powers of two
    hidden   = torch.where(subnormal, 0, 2**number_format.mantissa_bits        )
    field    = torch.where(subnormal, 0, exponent + number_format.exponent_bias).to(torch.int64)
    mantissa = (torch.ldexp(size, number_format.mantissa_bits - exponent) - hidden).to(torch.int64)
    code     = (field << number_format.mantissa_bits) | mantissa
    
    # Put the sign bit on top for signed formats
    if not number_format.signed:
        return code
    return code | (torch.signbit(values).to(torch.int64) << (number_format.bits - 1))


def fp_decode(codes: torch.Tensor, number_format: NumberFormat) -> torch.Tensor:
    '''
    #* Turn FP bit codes back into float32 values.
    #! The top code of a nan_top format becomes NaN.
    '''
    
    # Take the fields out of the code
    mantissa_bits = number_format.mantissa_bits
    mantissa      = codes & (2**mantissa_bits - 1)
    field         = (codes >> mantissa_bits) & (2**number_format.exponent_bits - 1)
    
    # Subnormals have no hidden 1, and use the lowest exponent
    hidden   = torch.where(field > 0, 2**mantissa_bits, 0)
    exponent = torch.clamp(field - number_format.exponent_bias, min=1 - number_format.exponent_bias)
    value    = torch.ldexp((hidden + mantissa).to(torch.float32), (exponent - mantissa_bits).to(torch.int32))
    
    # Add the sign, and turn the NaN code into NaN
    if number_format.signed:
        value = torch.where(((codes >> (number_format.bits - 1)) & 1) == 1, -value, value)
    if number_format.nan_top:
        top   = (field == 2**number_format.exponent_bits - 1) & (mantissa == 2**mantissa_bits - 1)
        value = torch.where(top, torch.nan, value)
    return value


def fp_values(number_format: NumberFormat) -> torch.Tensor:
    '''
    #* Return every value that the FP format can hold, sorted and without NaN (-0 and 0 count as one value).
    #? Tests and lookup tables use this.
    '''
    
    values = fp_decode(torch.arange(2**number_format.bits), number_format) + 0.0
    return torch.unique(values[~torch.isnan(values)])


def _fp_exponent(x: torch.Tensor, number_format: NumberFormat) -> torch.Tensor:
    '''
    #* Return floor(log2(|x|)) for each value, but not lower than the lowest exponent of the format.
    #? fp_step() and fp_encode() use this.
    '''
    
    _, exponent = torch.frexp(x)
    return torch.clamp(exponent - 1, min=1 - number_format.exponent_bias)
