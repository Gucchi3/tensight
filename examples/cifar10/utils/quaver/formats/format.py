# @file format.py
# @brief The number formats by name, and the shared rounding onto them
# @author Kohei Yamaguchi, Claude

##########################################################
'''
Add a number format
1. Make the format with integer_format() or fp_format().
2. Add "name: format" to FORMAT_REGISTRY.
'''
##########################################################

import re
import torch

from .device  import device_define
from .fp      import fp_decode, fp_encode, fp_format, fp_step
from .integer import integer_decode, integer_encode, integer_format
from .round   import build_round
from .types   import NumberFormat, RoundMode

# Formats of values: int2 to int8, uint2 to uint8, fp4, and ufp4. Formats of block scales (BlockScaler): ue4m3 (NVFP4), uint6, and uint10 (integer block scales)
FORMAT_REGISTRY: dict[str, NumberFormat] = {
    "int2"  : integer_format(2 , signed=True ),
    "int4"  : integer_format(4 , signed=True ),
    "int8"  : integer_format(8 , signed=True ),
    "uint2" : integer_format(2 , signed=False),
    "uint4" : integer_format(4 , signed=False),
    "uint6" : integer_format(6 , signed=False),
    "uint8" : integer_format(8 , signed=False),
    "uint10": integer_format(10, signed=False),
    "fp4"   : fp_format(2, 1, signed=True , nan_top=False),
    "ufp4"  : fp_format(2, 2, signed=False, nan_top=False),
    "ue4m3" : fp_format(4, 3, signed=False, nan_top=True , name="ue4m3"),
}



def build_format(name: str) -> NumberFormat:
    '''
    #* Return the number format for the name.
    #? INT formats of any width also work by name (int32 and int62 for the accumulators, uint1), so the device ops can get every format as a name.
    #! Add new formats to FORMAT_REGISTRY.
    '''
    
    if name in FORMAT_REGISTRY:
        return FORMAT_REGISTRY[name]
    match = re.fullmatch(r"(u?)int([1-9][0-9]*)", name)
    if match is None:
        raise KeyError(f"Unknown number format {name!r}. Available: {', '.join(FORMAT_REGISTRY)}, or intN and uintN of any width")
    return integer_format(int(match.group(2)), signed=match.group(1) == "")


def format_round_mode(number_format: NumberFormat, mode: RoundMode | None = None) -> RoundMode:
    '''
    #* Return mode, or the usual rounding of the format when mode is None: half_even for FP (like IEEE, OCP MX, and NVFP4), and half_away for INT (like TFLite).
    #? Quantizer and the samples use this for their default rounding.
    '''
    
    if mode is not None:
        return mode
    return "half_even" if number_format.kind == "fp" else "half_away"


def format_quantize(x: torch.Tensor, number_format: NumberFormat, mode: RoundMode, zero_point: torch.Tensor | None = None) -> torch.Tensor:
    '''
    #* Round x to the nearest value of the format, add the zero point, and stop at the ends of the range.
    #? INT and FP use this same code. Only the grid step is different: 1 for INT, and the step at each exponent for FP.
    #! Pass x that is already divided by the scale. Rounding comes before the zero point, like TFLite and the devices.
    '''
    
    # Round on the grid of the format
    round_fn = build_round(mode)
    if number_format.kind == "int":
        grid = round_fn(x)
    else:
        step = fp_step(x, number_format)
        grid = round_fn(x / step) * step
    
    # Add the zero point, and stop at the ends of the range
    if zero_point is not None:
        grid = grid + zero_point
    return grid.clamp(number_format.min_value, number_format.max_value)


@device_define("quantize")
def format_codes(x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, number_format: str, mode: RoundMode) -> torch.Tensor:
    '''
    #* Device op "quantize": the integer codes (int64) of x / scale on the format, with the zero point, stopped at the ends of the range.
    #? Quantizer.encode() calls it as torch.ops.quaver.quantize. In an exported model it quantizes the input. The weights are folded into codes before the export.
    #! FP codes are integers on the grid of the format (value / step_min), so FP4 1.5 is 3, as in every integer op of quaver.
    '''
    
    named = build_format(number_format)
    return format_to_integer(format_quantize(x.float() / scale, named, mode, zero_point), named)


def format_encode(x: torch.Tensor, number_format: NumberFormat, mode: RoundMode) -> torch.Tensor:
    '''
    #* Round x to the format, and return the codes for storage as int64: INT values, or FP bit codes.
    #? Use this to store or export weights.
    '''
    
    grid = format_quantize(x, number_format, mode)
    if number_format.kind == "int":
        return integer_encode(grid, number_format)
    return fp_encode(grid, number_format)


def format_decode(codes: torch.Tensor, number_format: NumberFormat) -> torch.Tensor:
    '''
    #* Turn the codes for storage back into float32 values.
    '''
    
    if number_format.kind == "int":
        return integer_decode(codes, number_format)
    return fp_decode(codes, number_format)


def format_to_integer(values: torch.Tensor, number_format: NumberFormat) -> torch.Tensor:
    '''
    #* Turn grid values into integers for exact integer math: value / step_min as int64.
    #? INT values stay the same. FP4 values are doubled (the step is 0.5), so 1.5 becomes 3.
    '''
    
    return torch.round(values / number_format.step_min).to(torch.int64)
