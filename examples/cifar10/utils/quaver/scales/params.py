# @file params.py
# @brief Move between real values and integer codes, and the float type of the training-mode math
# @author Kohei Yamaguchi, Claude

import torch

from ..formats import RoundMode, build_format, device_define, format_quantize, format_to_integer, ste_attach
from .types    import BlockScale, QuantParams

MATH_DTYPES: dict[str, torch.dtype] = {
    "train": torch.float64,
    "test" : torch.float64,
}



def params_math_dtype() -> torch.dtype:
    '''
    #* Return the float type of the training-mode math now: the "train" type while gradients are on, and the "test" type while they are off.
    #? The fake ops call this. precision_set_math() changes MATH_DTYPES (both float64 at the start). The integer ops always use float64.
    #! torch.is_grad_enabled() only reads a flag on the CPU, so this makes no GPU sync. A CUDA graph keeps the type it was captured with.
    '''
    
    return MATH_DTYPES["train" if torch.is_grad_enabled() else "test"]


def params_codes(x: torch.Tensor, params: QuantParams, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    '''
    #* Get the integer codes (int64) back from fake-quantized values.
    #? The fake ops use this to run the same integer math as the integer mode. They pass the type of params_math_dtype().
    #! In float64 this is exact if the float error of x is less than half a code step (turn off TF32). float32 can be off by one for large accumulators.
    '''
    
    return torch.round(x.detach().to(dtype) / params.code_scale.to(dtype) + params.zero_point.to(dtype)).to(torch.int64)


def params_values(codes: torch.Tensor, params: QuantParams, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    '''
    #* Turn integer codes back into real float32 values.
    #? dtype is the type of the math. float32 gives the same values as float64 while |code - zero point| is below 2^24.
    '''
    
    return ((codes.to(dtype) - params.zero_point.to(dtype)) * params.code_scale.to(dtype)).float()


def params_unit(params: QuantParams) -> torch.Tensor:
    '''
    #* Return the real value of one step of the integers that exact sums use: the code scale, or for block-scaled tensors step_min x step of the block scales x tensor scale.
    #? The integers are params_integers(). The blocks of a tensor share this one unit, so a sum over many blocks has one scale.
    '''
    
    if params.block is None:
        return params.code_scale
    return params.block.tensor_scale * (params.number_format.step_min * params.block.scale_format.step_min)


def params_integers(codes: torch.Tensor, params: QuantParams) -> torch.Tensor:
    '''
    #* Return the integers that exact sums use (int64): code - zero point, and for block-scaled tensors also times the code of the block scale.
    #? Real value = integer x params_unit(params). With the block scale inside, values of different blocks can be added as they are.
    '''
    
    centered = codes - params.zero_point.to(torch.int64)
    if params.block is None:
        return centered
    return centered * params.block.codes


def params_bits(params: QuantParams) -> int:
    '''
    #* Return the number of bits of the largest |integer| of params_integers(), from the formats only (no data).
    #? The exact sums use this to see, before running, how wide they get.
    '''
    
    # Plain tensors: the codes minus the zero point fit in the span of the format
    number_format = params.number_format
    largest       = round((number_format.max_value - number_format.min_value) / number_format.step_min)
    if params.block is None:
        return largest.bit_length()
    
    # Block-scaled tensors: the largest |code| times the largest code of the block scales
    scale_format = params.block.scale_format
    top          = round(max(-number_format.min_value, number_format.max_value) / number_format.step_min)
    return (top * round(scale_format.max_value / scale_format.step_min)).bit_length()


def params_named(zero_point: torch.Tensor, number_format: str, block_codes: torch.Tensor | None = None, scale_format: str = "") -> QuantParams:
    '''
    #* Make the parameters of codes from the zero point and the format name (and the codes of the block scales), with scales of 1.
    #? The device ops (torch.ops.quaver.*) get plain tensors and names, and rebuild the parameters with this for the integer helpers.
    #! The scales are 1, so use the result only where the scales do not matter: exact sums (params_integers, params_bits) and rounding in code units (params_round).
    '''
    
    one   = torch.ones((), device=zero_point.device)
    block = None if block_codes is None else BlockScale(block_codes, build_format(scale_format), one)
    return QuantParams(one, zero_point, build_format(number_format), block)


def params_note(codes: torch.Tensor, params: QuantParams) -> torch.Tensor:
    '''
    #* While torch.export runs, put the format, the scale, and the zero point of the codes into the graph (the op "note"), and give the codes back. Otherwise return the codes as they are.
    #? The integer functions of the ops call this on their outputs. model_export() writes the note on the value of the codes and leaves the node out, so the note costs nothing on the device.
    #! A block-scaled tensor notes its tensor scale, and an output of NEQ its integer p. The other scales are noted as they are.
    '''
    
    if not torch.compiler.is_exporting():
        return codes
    if params.block is not None:
        return torch.ops.quaver.note(codes, params.block.tensor_scale, params.zero_point, params.number_format.name, "tensor_scale")
    if params.peak is not None:
        return torch.ops.quaver.note(codes, params.peak, params.zero_point, params.number_format.name, "peak")
    return torch.ops.quaver.note(codes, params.scale, params.zero_point, params.number_format.name, "scale")


@device_define("note")
def params_note_device(codes: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, number_format: str, scale_name: str) -> torch.Tensor:
    '''
    #* The op "note": a copy of the codes, with their format, their scale (named scale_name), and their zero point as arguments.
    #? params_note() calls it only while torch.export runs. The export reads it and removes it.
    '''
    
    return codes.clone()


def params_round(value: torch.Tensor, params: QuantParams, mode: RoundMode) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    #* Round values in code units (before the zero point) to codes of the format, and say which values were inside the range.
    #? Every op of the integer mode ends here, so INT and FP outputs use the same path. mode is for INT formats. FP formats always round half to even, like FP hardware (IEEE, OCP MX).
    #! Pass exact values, such as sums and power-of-two products of integers (in float64 they are exact up to 2^53). The math keeps the float type of value.
    '''
    
    number_format = params.number_format
    grid          = value * number_format.step_min
    zero          = params.zero_point.to(value.dtype)
    inside        = (grid + zero >= number_format.min_value) & (grid + zero <= number_format.max_value)
    return format_to_integer(format_quantize(grid, number_format, mode if number_format.kind == "int" else "half_even", zero), number_format), inside


def params_attach(reference: torch.Tensor, codes: torch.Tensor, params: QuantParams, inside: torch.Tensor, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    '''
    #* Return the real values of the codes in the forward pass, and the gradients of fake-quantizing reference in the backward pass.
    #? The fake ops call this after the integer math. reference gets a straight-through gradient inside the range. A learned scale and zero point get the LSQ gradients.
    #! The forward value is exactly params_values(codes, params, dtype). Pass a learned scale with its LSQ gradient scale on it (Quantizer.measure() does this).
    '''
    
    # reference: a straight-through gradient inside the range
    attached = ste_attach(reference, params_values(codes, params, dtype), inside)
    if not (params.scale.requires_grad or params.zero_point.requires_grad):
        return attached
    
    # A learned scale and zero point: d/dscale = code - reference / scale inside the range, and the end of the range outside (LSQ), with the codes of the integer math
    number_format = params.number_format
    scaled        = reference.detach() / params.scale + params.zero_point
    clipped       = torch.where(inside, scaled, scaled.clamp(number_format.min_value, number_format.max_value).detach())
    grid          = codes.to(scaled.dtype) * number_format.step_min
    learned       = (clipped + (grid - clipped).detach() - params.zero_point) * params.scale
    return attached + (learned - learned.detach())
