# @file collect.py
# @brief Collect the quantizers and the scalers in a model
# @author Kohei Yamaguchi, Claude

from torch import nn

from ..quantizers import Quantizer



def collect_quantizers(model: nn.Module) -> list[tuple[str, Quantizer]]:
    '''
    #* Collect every Quantizer in the model, with its name.
    '''
    
    return [(name, module) for name, module in model.named_modules() if isinstance(module, Quantizer)]


def collect_scalers[T: nn.Module](model: nn.Module, kind: type[T]) -> list[tuple[str, T]]:
    '''
    #* Collect every scaler of one class in the model, with its name.
    #? kind is the class, such as RangeScaler or LsqScaler. Pass nn.Module to get every scaler (every module with a learnable attribute).
    '''
    
    return [(name, module) for name, module in model.named_modules() if isinstance(module, kind) and hasattr(module, "learnable")]
