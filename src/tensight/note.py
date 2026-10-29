# @file note.py
# @brief Record a tensor in the middle of a forward, with a name (02a API-06)
# @author Kohei Yamaguchi, Claude

import re

from .adapters import registry_adapters
from .core     import message_warn
from .observe  import capture_active, glob_compile, layout_choose, walk_value
from .run      import run_current
from .stats    import pattern_parse
from .types    import MapPattern, PointInfo



NOTE_NAME = re.compile(r"[A-Za-z0-9._-]{1,64}")

_names: set[str] = set()



def note_tensor[T](name: str, value: T, *, map: str | None = None) -> T:
    '''
    #* Record a tensor, or a nested value of tensors, as the point "name:note", and return the value as it is (02a API-06).
    #! In steps that are not recorded, it only checks the name once and returns (NFR-001).
    #! While a CUDA graph is captured or torch.compile compiles, it returns at once and checks nothing: dynamo would make what it reads into guards (ADR-016).
    '''
    
    # Do nothing while a CUDA graph is captured or torch.compile compiles
    if capture_active():
        return value
    
    # Check the name once, and the map pattern (02a API-06)
    if not isinstance(name, str):
        raise TypeError("引数 name は str にしてください。渡された値: " + repr(name))
    if name not in _names:
        if NOTE_NAME.fullmatch(name) is None:
            raise ValueError("引数 name は 1 文字以上 64 文字以下の、英字・数字・.・-・_ だけの文字列にしてください。渡された値: " + repr(name))
        _names.add(name)
    if map is not None and not isinstance(map, str):
        raise TypeError("引数 map は str か None にしてください。渡された値: " + repr(map))
    spec = pattern_parse(map) if map is not None else None
    
    # Do nothing outside a recorded step
    state = run_current()
    if state is None:
        message_warn("MSG-006", function="note_tensor")
        return value
    if not state.recording:
        return value
    
    # Hand each tensor of the value to the collector, with its map pattern
    try:
        for leaf in walk_value(value, name + ":note", registry_adapters()):
            point = name + ":note" + leaf.suffix
            found = spec if spec is not None else next((given for key, given in state.settings.maps if glob_compile(key).fullmatch(point)), None)
            if found == "auto":
                pattern, warn = layout_choose(None, "note", leaf.tensor.ndim)
                if warn:
                    message_warn("MSG-013", point=point, layer="なし", pattern=pattern.text if pattern is not None else "なし")
            else:
                pattern = found if isinstance(found, MapPattern) else None
            state.collector.add_tensor(PointInfo(point, name, "note", None), leaf.tensor, pattern, leaf.quant)
    except Exception as error:
        message_warn("MSG-002", point=name + ":note", error=error)
    return value
