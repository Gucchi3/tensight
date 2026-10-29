# @file pack.py
# @brief Pack codes into bytes like the memory of the device: 2-bit and 4-bit codes from the low bits up, wider codes little-endian, and FP codes as their bit patterns
# @author Kohei Yamaguchi, Claude

from typing import Any

import torch

from ..formats import build_format, fp_decode, fp_encode

# The widths in memory: 2 and 4 bits are packed into bytes, and every other width takes the next of 8, 16, 32, and 64 bits (int6 in 8 bits, uint10 in 16)
PACK_WIDTHS = (2, 4, 8, 16, 32, 64)



def pack_codes(codes: torch.Tensor, number_format: str) -> tuple[bytes, dict[str, Any]]:
    '''
    #* Pack codes (in the order of their elements) into bytes, and say how they are stored: the encoding and the width in bits.
    #? INT codes are two's complement in their width ("int"). FP codes (grid integers in quaver) are stored as their bit patterns: E2M1 for fp4, E2M2 for ufp4, E4M3 for ue4m3 ("fp").
    #? "float32" and "float64" store floats as IEEE 754 ("ieee754"). 2-bit and 4-bit values fill each byte from the low bits up: the earlier element is in the low bits.
    #! Values out of the range of the format raise ValueError. Needs numpy (the "export" extra).
    '''
    
    # numpy writes the bytes in a fixed order (little-endian), so the file does not depend on the computer
    import numpy
    
    # Floats as IEEE 754
    flat = codes.detach().cpu().reshape(-1)
    if number_format in ("float32", "float64"):
        bits = 32 if number_format == "float32" else 64
        return flat.double().numpy().astype(f"<f{bits // 8}").tobytes(), {"encoding": "ieee754", "bits": bits}
    
    # The integers to store: INT codes as they are, and FP codes as their bit patterns (sign, exponent, mantissa)
    named = build_format(number_format)
    if named.kind == "fp":
        stored = fp_encode(flat.double() * named.step_min, named)
        info   = {"encoding": "fp", "signed": named.signed, "exponent_bits": named.exponent_bits, "mantissa_bits": named.mantissa_bits, "exponent_bias": named.exponent_bias}
        low    = 0
        high   = 2**named.bits - 1
    else:
        stored = flat.long()
        info   = {"encoding": "int", "signed": named.signed}
        low    = round(named.min_value)
        high   = round(named.max_value)
    if stored.numel() and (int(stored.min()) < low or int(stored.max()) > high):
        raise ValueError(f"Codes out of the range of {number_format}: {int(stored.min())} to {int(stored.max())}.")
    if named.kind == "fp" and not torch.equal(fp_decode(stored, named).double() / named.step_min, flat.double()):
        raise ValueError(f"Codes that are not on the grid of {number_format}.")
    
    # Wide values: little-endian, two's complement for signed INT
    width = next(width for width in PACK_WIDTHS if width >= named.bits and (width >= 8 or width == named.bits))
    if width >= 8:
        kind = "i" if info["signed"] and named.kind == "int" else "u"
        return stored.numpy().astype(f"<{kind}{width // 8}").tobytes(), {**info, "bits": width}
    
    # Narrow values: the low bits of each value (two's complement), several in each byte from the low bits up
    count  = 8 // width
    values = (stored & (2**width - 1)).numpy().astype("uint8")
    values = numpy.pad(values, (0, -len(values) % count)).reshape(-1, count)
    packed = numpy.zeros(len(values), dtype="uint8")
    for index in range(count):
        packed |= values[:, index] << (index * width)
    return packed.tobytes(), {**info, "bits": width}
