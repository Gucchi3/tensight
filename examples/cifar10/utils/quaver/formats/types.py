# @file types.py
# @brief Types for the number formats
# @author Kohei Yamaguchi, Claude

from dataclasses import dataclass
from typing      import Literal

RoundMode = Literal["half_away", "half_even", "half_up", "floor"]



@dataclass(frozen=True)
class NumberFormat:
    '''
    #* Hold the details of one number format (INT or FP), on the grid after the value is divided by the scale.
    #? Make one with integer_format() or fp_format(), or get one by name with build_format().
    '''
    
    name         : str
    kind         : str
    bits         : int
    signed       : bool
    exponent_bits: int
    mantissa_bits: int
    exponent_bias: int
    nan_top      : bool
    min_value    : float
    max_value    : float
    step_min     : float
