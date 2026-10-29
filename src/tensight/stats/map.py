# @file map.py
# @brief The map of absolute values (FR-035, ADR-009)

import torch

from .types import MapPlan



MAP_SIZE = 64



def map_build(values: torch.Tensor, plan: MapPlan) -> torch.Tensor:
    '''
    #* Make the map: the largest absolute value of the elements that each cell covers, at most MAP_SIZE x MAP_SIZE cells.
    #? Called from summary_tensor(). It runs on the device of the values and does not wait for the GPU.
    #! NaN and Inf count as 0. Group i of n holds the indices from floor(i x n / 64) up to, but not including, floor((i + 1) x n / 64).
    '''
    
    # Take the absolute values, and the largest one over the dimensions that are not on the right side
    cells = torch.where(torch.isfinite(values), values.abs(), torch.zeros((), dtype=values.dtype, device=values.device))
    if plan.reduce:
        cells = torch.amax(cells, dim=plan.reduce)
    
    # Put the rows and the columns in the order of the pattern
    kept  = [dim for dim in range(values.ndim) if dim not in plan.reduce]
    cells = cells.permute([kept.index(dim) for dim in (*plan.rows, *plan.cols)]).reshape(plan.row_size, plan.col_size)
    
    # Group consecutive rows, then columns, into MAP_SIZE groups with the largest value of each group
    for axis, size in ((0, plan.row_size), (1, plan.col_size)):
        if size <= MAP_SIZE:
            continue
        group       = ((torch.arange(size, device=cells.device) + 1) * MAP_SIZE + size - 1) // size - 1
        index       = group.reshape(-1, 1).expand(-1, cells.shape[1]) if axis == 0 else group.reshape(1, -1).expand(cells.shape[0], -1)
        shape       = list(cells.shape)
        shape[axis] = MAP_SIZE
        cells       = torch.zeros(shape, dtype=cells.dtype, device=cells.device).scatter_reduce(axis, index, cells, reduce="amax", include_self=True)
    return cells.to(torch.float32)
