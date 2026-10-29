# @file group.py
# @brief Split tensors into groups that share one scale
# @author Kohei Yamaguchi, Claude

import math
import torch

from .types import Granularity



def group_count(shape: torch.Size, granularity: Granularity) -> int:
    '''
    #* Return how many scales a tensor of this shape gets.
    '''
    
    if granularity.kind == "tensor":
        return 1
    if granularity.kind == "channel":
        return shape[granularity.axis]
    return math.prod(shape) // granularity.block_size


def group_split(x: torch.Tensor, granularity: Granularity) -> torch.Tensor:
    '''
    #* Reshape x into (groups, values), so that each row shares one scale.
    #? Tensor, channel, and block granularities all use this one function.
    #! For blocks, the size of the axis must be a multiple of block_size.
    '''
    
    if granularity.kind == "tensor":
        return x.reshape(1, -1)
    if granularity.kind == "channel":
        return x.movedim(granularity.axis, 0).reshape(x.shape[granularity.axis], -1)
    return x.movedim(granularity.axis, -1).reshape(-1, granularity.block_size)


def group_merge(values: torch.Tensor, shape: torch.Size, granularity: Granularity) -> torch.Tensor:
    '''
    #* Spread one value for each group back into a tensor that broadcasts against the shape.
    #? Tensor gives a 0-dim value, and channel gives a small shape. A block over the whole axis gives one value for each token.
    #! Smaller blocks give the full shape.
    '''
    
    # One value in total (0-dim, so it fits any reshape of x), or one for each channel
    if granularity.kind == "tensor":
        return values.reshape(())
    if granularity.kind == "channel":
        view                   = [1] * len(shape)
        view[granularity.axis] = shape[granularity.axis]
        return values.reshape(view)
    
    # Blocks: one value for each block, spread over the block unless the block covers the whole axis
    axis   = granularity.axis % len(shape)
    others = [size for index, size in enumerate(shape) if index != axis]
    blocks = shape[axis] // granularity.block_size
    spread = values.reshape(*others, blocks, 1)
    if blocks > 1:
        spread = spread.expand(*others, blocks, granularity.block_size)
    return spread.reshape(*others, -1).movedim(-1, axis)
