# @file minmax.py
# @brief Measure the range of values in each group
# @author Kohei Yamaguchi, Claude

from fractions import Fraction

import math
import torch

from .group import group_split
from .types import Granularity



def minmax_measure(x: torch.Tensor, granularity: Granularity, percentile: float | None = None, symmetric: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    #* Measure the range of each group, and return it as flat (low, high) tensors.
    #? This one function covers the maximum and the percentile, symmetric and asymmetric, and every granularity.
    #! Symmetric measures |x| and returns (-high, high). There is no gradient.
    '''
    
    # Measure |x| for symmetric ranges
    groups = group_split(x.detach(), granularity)
    if symmetric:
        groups = groups.abs()
    
    # Take the full range
    if percentile is None:
        low, high = torch.aminmax(groups, dim=1)
        return (-high, high) if symmetric else (low, high)
    
    # Or take the percentiles by the nearest rank: the high side at percentile, the low side at 100 - percentile (the rank uses exact fractions)
    count = groups.shape[1]
    rank  = max(1, math.ceil(Fraction(str(percentile)) / 100 * count))
    high  = groups.kthvalue(rank, dim=1).values
    if symmetric:
        return -high, high
    return groups.kthvalue(count - rank + 1, dim=1).values, high
