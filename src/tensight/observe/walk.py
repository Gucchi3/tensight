# @file walk.py
# @brief How nested values are walked to find tensors, and the converters of unknown types (FR-002, 02a 4章)

from collections.abc import Callable, Mapping, Sequence
from typing          import Any

import dataclasses
import torch

from ..core  import message_warn
from ..types import Adapter, QuantValue
from .types  import WalkLeaf



CONVERTER_REGISTRY: dict[type, Callable[[Any], object]] = {}

WALK_DEPTH   = 8
WALK_LIMIT   = 256
SILENT_TYPES = (int, float, complex, bool, str, bytes, type(None))



def walk_add_converter(cls: type, converter: Callable[[Any], object]) -> None:
    '''
    #* Register a converter for a type; the same type is replaced (IF-10).
    #? Called from converter_register(), which checks the arguments.
    '''
    
    CONVERTER_REGISTRY[cls] = converter
    return None


def walk_value(value: object, base: str, adapters: Sequence[Adapter]) -> list[WalkLeaf]:
    '''
    #* Find every tensor in a nested value, depth first and from the front (02a 4章 table of nested values).
    #? Called from the hooks of ModelObserver and from note_tensor(). base is the first half of the point name, for example "attn:output".
    #! It never raises. Converters and readers that fail show a warning.
    '''
    
    # Walk with a stack of (value, suffix, depth), starting from the value itself
    leaves = list[WalkLeaf]()
    names  = dict[str, int]()
    stack  = [(value, "", 0)]
    while stack:
        item, suffix, depth = stack.pop()
        if depth > WALK_DEPTH:
            message_warn("MSG-014", point=base, reason="入れ子が 8 段より深い")
            continue
        
        # 1. A registered converter (also for the subclasses of its type)
        converter = next((CONVERTER_REGISTRY[kind] for kind in type(item).__mro__ if kind in CONVERTER_REGISTRY), None)
        if converter is not None:
            try:
                stack.append((converter(item), suffix, depth + 1))
            except Exception as error:
                message_warn("MSG-011", type=type(item).__module__ + "." + type(item).__qualname__, error=error)
            continue
        
        # 2. A value that a reader handles, and 3. a tensor
        quant: QuantValue | None = None
        for adapter in adapters:
            try:
                quant = adapter.read_value(item)
            except Exception as error:
                message_warn("MSG-003", adapter=adapter.name, error=error)
                quant = None
            if quant is not None:
                break
        
        # Keep the tensor itself, or the tensor that a reader got (its source, or else its values)
        tensor = item if isinstance(item, torch.Tensor) else None if quant is None else quant.source if quant.source is not None else quant.values
        if tensor is not None:
            if len(leaves) >= WALK_LIMIT:
                message_warn("MSG-014", point=base, reason="テンソルが 256 個を超えた")
                break
            number        = names.get(suffix, 0) + 1
            names[suffix] = number
            leaves.append(WalkLeaf(suffix if number == 1 else suffix + "#" + str(number), tensor, quant))
            continue
        
        # 4 to 7. Containers: their children, named by field, key or position
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            children = [(getattr(item, field.name), "[" + field.name + "]") for field in dataclasses.fields(item)]
        elif isinstance(item, tuple) and hasattr(type(item), "_fields"):
            children = [(getattr(item, name), "[" + name + "]") for name in getattr(type(item), "_fields")]
        elif isinstance(item, Mapping):
            children = [(child, "[" + str(key) + "]") for key, child in item.items()]
        elif isinstance(item, (tuple, list)):
            children = [(child, "[" + str(position) + "]") for position, child in enumerate(item)]
        
        # 8. Numbers, text and None are skipped quietly, and 9. other types with a warning
        elif isinstance(item, SILENT_TYPES):
            continue
        else:
            message_warn("MSG-010", type=type(item).__module__ + "." + type(item).__qualname__)
            continue
        stack.extend((child, suffix + key, depth + 1) for child, key in reversed(children))
    return leaves
