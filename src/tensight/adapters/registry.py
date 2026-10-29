# @file registry.py
# @brief The readers of quantization libraries, in the order to try them (FR-017)

from ..types         import Adapter
from .quaver_adapter import QuaverAdapter
from .torch_adapter  import TorchAdapter



ADAPTER_REGISTRY: list[Adapter]       = []
BUILTIN_ADAPTERS: tuple[Adapter, ...] = (QuaverAdapter(), TorchAdapter())



def registry_add(adapter: Adapter) -> None:
    '''
    #* Add a reader that the user made, or replace the one with the same name (02a API-07).
    #! adapter_register() has checked its shape.
    '''
    
    for i, old in enumerate(ADAPTER_REGISTRY):
        if old.name == adapter.name:
            ADAPTER_REGISTRY[i] = adapter
            return None
    ADAPTER_REGISTRY.append(adapter)
    return None


def registry_adapters() -> list[Adapter]:
    '''
    #* Return the readers in the order to try them: the user's ones first, then quaver, then PyTorch.
    '''
    
    return [*ADAPTER_REGISTRY, *BUILTIN_ADAPTERS]
