# @file converter.py
# @brief Register a converter that takes tensors out of values of an unknown type (02a API-10)
# @author Kohei Yamaguchi, Claude

from collections.abc import Callable
from typing          import Any

import torch

from .observe import walk_add_converter



def converter_register(cls: type, converter: Callable[[Any], object]) -> None:
    '''
    #* Register a converter for a type and its subclasses; the same type is replaced (02a API-10, IF-10).
    #! torch.Tensor itself cannot be registered: tensors are recorded without a converter.
    '''
    
    if not isinstance(cls, type):
        raise TypeError("cls には型を渡してください。渡された値: " + repr(cls))
    if not callable(converter):
        raise TypeError("converter には呼び出せる関数を渡してください。渡された値: " + repr(converter))
    if cls is torch.Tensor:
        raise ValueError("torch.Tensor は、変換の関数なしで記録します（torch.Tensor を継承した型は登録できます）")
    walk_add_converter(cls, converter)
    return None
