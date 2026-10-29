# @file round.py
# @brief Exact rounding functions
# @author Kohei Yamaguchi, Claude

##########################################################
'''
Add a rounding mode
1. Write round_xxx(x) to return the rounded tensor.
2. Add "name: round_xxx" to ROUND_REGISTRY.
3. Add the name to RoundMode in types.py.
'''
##########################################################

from collections.abc import Callable

import torch



def round_half_away(x: torch.Tensor) -> torch.Tensor:
    '''
    #* Round to the nearest integer. A half goes away from zero.
    #! This is exact for every float. floor(|x| + 0.5) is not exact, so do not use it.
    '''
    
    whole = torch.trunc(x)
    return whole + torch.trunc(2.0 * (x - whole))


def round_half_even(x: torch.Tensor) -> torch.Tensor:
    '''
    #* Round to the nearest integer. A half goes to the even integer.
    '''
    
    return torch.round(x)


def round_half_up(x: torch.Tensor) -> torch.Tensor:
    '''
    #* Round to the nearest integer. A half goes up (toward +inf).
    #! This is exact for every float. floor(x + 0.5) is not exact, so do not use it.
    '''
    
    floor = torch.floor(x)
    return floor + (x - floor >= 0.5).to(x.dtype)


def round_floor(x: torch.Tensor) -> torch.Tensor:
    '''
    #* Round down, like an arithmetic right shift.
    '''
    
    return torch.floor(x)


ROUND_REGISTRY: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "half_away": round_half_away,
    "half_even": round_half_even,
    "half_up"  : round_half_up,
    "floor"    : round_floor,
}


def build_round(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    '''
    #* Return the rounding function for the name.
    #! Add new modes to ROUND_REGISTRY.
    '''
    
    if name not in ROUND_REGISTRY:
        raise KeyError(f"Unknown rounding mode {name!r}. Available: {', '.join(ROUND_REGISTRY)}")
    return ROUND_REGISTRY[name]
