# @file switch.py
# @brief Turn quantization and the integer mode on or off
# @author Kohei Yamaguchi, Claude

from torch import nn

from .collect import collect_quantizers



def switch_quantizers(model: nn.Module, enabled: bool) -> int:
    '''
    #* Turn quantization on or off in every Quantizer, and put every layer with a float_mode flag into the float mode while it is off. Returns how many quantizers changed.
    #? When it is off, the layers compute in float and the quantizers still update their scalers, like the observers of PyTorch (calibrate_model() uses this).
    #! Capture CUDA graphs again after you switch. The layers have no common class, so the flag is set with setattr().
    '''
    
    # Turn every quantizer on or off
    quantizers = collect_quantizers(model)
    for _, quantizer in quantizers:
        quantizer.enabled = enabled
    
    # The layers compute in float while quantization is off
    for module in model.modules():
        if hasattr(module, "float_mode"):
            setattr(module, "float_mode", not enabled)
    return len(quantizers)


def switch_integer_mode(model: nn.Module, enabled: bool) -> int:
    '''
    #* Put every layer with an integer_mode flag into the integer mode, or back into the training mode. Returns how many changed.
    #? In the integer mode, the layers take and return integer codes, and compute exactly like the device.
    #! The layers have no common class, so the flag is set with setattr().
    '''
    
    layers = [module for module in model.modules() if hasattr(module, "integer_mode")]
    for layer in layers:
        setattr(layer, "integer_mode", enabled)
    return len(layers)
