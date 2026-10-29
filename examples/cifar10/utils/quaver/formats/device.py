# @file device.py
# @brief Register the integer steps of the device as torch ops (torch.ops.quaver.*), so that torch.export keeps each step as one node
# @author Kohei Yamaguchi, Claude

##########################################################
'''
Add a device op
1. Write xxx_device(...) next to its op: the integer step that the device runs, on plain tensors, ints, and names (no QuantParams).
2. Put @device_define("name") on it.
3. Call it as torch.ops.quaver.name(...) from the integer function of the op, so torch.export sees one node.
'''
##########################################################

from collections.abc import Callable
from typing          import Any, Literal, get_origin

import torch
import types

# The library of the quaver ops. CompositeExplicitAutograd runs the same Python code on every device, and on the fake tensors of torch.export
DEVICE_LIBRARY = torch.library.Library("quaver", "DEF")



def device_define(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    '''
    #* Return a decorator that registers a function as the op torch.ops.quaver.<name>, and gives the function back unchanged.
    #? The schema comes from the type hints. Literal names (RoundMode and the like) become str, since torch ops take plain strings.
    #! Call the op through torch.ops.quaver.<name>. Calling the function itself also works, but torch.export then sees the steps inside it.
    '''
    
    def register(function: Callable[..., Any]) -> Callable[..., Any]:
        '''
        #* Register function under the name, with Literal hints read as str.
        '''
        
        plain                 = types.FunctionType(function.__code__, function.__globals__, function.__name__, function.__defaults__, function.__closure__)
        plain.__annotations__ = {key: str if get_origin(value) is Literal else value for key, value in function.__annotations__.items()}
        DEVICE_LIBRARY.define(name + torch.library.infer_schema(plain, mutates_args=()))
        DEVICE_LIBRARY.impl(name, function, "CompositeExplicitAutograd")
        return function
    
    return register
