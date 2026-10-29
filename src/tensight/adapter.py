# @file adapter.py
# @brief Add a reader of a quantization library (02a API-07)

from .adapters import registry_add
from .types    import Adapter



def adapter_register(adapter: Adapter) -> None:
    '''
    #* Add a reader; it is tried before the built-in readers, and replaces one with the same name (02a API-07).
    #! Raises TypeError that lists every name the reader lacks.
    '''
    
    missing  = [] if isinstance(getattr(adapter, "name", None), str) else ["name"]
    missing += [method for method in ("find_quantizers", "read_value", "read_quantizer") if not callable(getattr(adapter, method, None))]
    if missing:
        raise TypeError("読み取り部品に、次のものがありません: " + "、".join(missing))
    registry_add(adapter)
    return None
