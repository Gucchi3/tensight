# @file activation.py
# @brief ReLU and ReLU6 on integer codes
# @author Kohei Yamaguchi, Claude

import torch

from ..formats import device_define, ste_attach
from ..scales  import QuantParams, params_codes, params_math_dtype, params_note, params_values
from .types    import ActivationKind



def activation_fake(x: torch.Tensor, x_params: QuantParams, kind: ActivationKind) -> torch.Tensor:
    '''
    #* Training mode ReLU or ReLU6: clamp the codes like activation_integer, with straight-through gradients.
    #? kind is "none", "relu", or "relu6". The math uses the type of params_math_dtype().
    '''
    
    dtype         = params_math_dtype()
    zero          = x_params.zero_point.to(torch.int64)
    six           = zero + torch.round(6.0 / x_params.code_scale.double()).to(torch.int64) if kind == "relu6" else zero
    codes, inside = _activation_codes(params_codes(x, x_params, dtype), zero, six, kind)
    return ste_attach(x, params_values(codes, x_params, dtype), inside)


def activation_integer(q_x: torch.Tensor, x_params: QuantParams, kind: ActivationKind) -> torch.Tensor:
    '''
    #* Integer mode ReLU or ReLU6: clamp the codes between the code of 0 and the code of 6.
    #? The device op torch.ops.quaver.activation does the clamp. The code of 6 is static (found here, and given to the device). No activation keeps the codes, without an op.
    #? Either way the codes get a note of their parameters, so an exported model knows the scale of the accumulator.
    #! Use it on accumulators (INT32) before you requantize, so the limits are exact integers.
    '''
    
    if kind == "none":
        return params_note(q_x, x_params)
    zero = x_params.zero_point.to(torch.int64)
    six  = zero + torch.round(6.0 / x_params.code_scale.double()).to(torch.int64) if kind == "relu6" else zero
    return params_note(torch.ops.quaver.activation(q_x, zero, six, kind), x_params)


@device_define("activation")
def activation_device(q_x: torch.Tensor, zero_point: torch.Tensor, six: torch.Tensor, kind: ActivationKind) -> torch.Tensor:
    '''
    #* Device op "activation": ReLU clamps the codes at the code of 0 (the zero point), and ReLU6 also at the code of 6 (six).
    #? activation_integer() calls it as torch.ops.quaver.activation, for "relu" and "relu6". ReLU gets the zero point as six, and does not use it.
    '''
    
    codes, _ = _activation_codes(q_x, zero_point, six, kind)
    return codes


def activation_float(x: torch.Tensor, x_params: QuantParams, kind: ActivationKind) -> torch.Tensor:
    '''
    #* Float mode (calibration): ReLU or ReLU6 on the float values.
    '''
    
    if kind == "none":
        return x
    return x.clamp(0.0, 6.0 if kind == "relu6" else None)


def _activation_codes(codes: torch.Tensor, zero: torch.Tensor, six: torch.Tensor, kind: ActivationKind) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    #* Clamp the codes at the code of 0 (and at the code of 6 for ReLU6), and say which codes did not change.
    #? activation_fake() and activation_device() use this, so both modes always agree.
    '''
    
    # No activation keeps every code
    if kind == "none":
        return codes, torch.ones_like(codes, dtype=torch.bool)
    
    # Clamp at the code of 0, and at the code of 6 for ReLU6
    limited = torch.maximum(codes, zero)
    if kind == "relu6":
        limited = torch.minimum(limited, six)
    return limited, limited == codes
