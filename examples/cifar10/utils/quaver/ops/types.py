# @file types.py
# @brief Types for the quantized ops
# @author Kohei Yamaguchi, Claude

from collections.abc import Callable
from dataclasses     import dataclass
from typing          import Literal

import torch

from ..formats import RoundMode

ActivationKind = Literal["none", "relu", "relu6"]
NeqScope       = Literal["token", "tensor"]



@dataclass(frozen=True)
class RequantizeStyle:
    '''
    #* Describe how a device rescales an accumulator: the limit of the multiplier, the limit of the shift, and the rounding.
    #? multiplier_max = 0 means "exact": keep the real ratio in float64, not a multiplier and a shift.
    #! The product (accumulator x multiplier) is kept at full width, like 32 + 16 bits for a 16-bit multiplier.
    '''
    
    name          : str
    multiplier_max: int
    shift_max     : int
    round_mode    : RoundMode


@dataclass(frozen=True)
class FixedPoint:
    '''
    #* Hold the multiplier and the shift that stand for a ratio of scales: ratio = multiplier / 2^shift.
    #? The "exact" style keeps the ratio itself in multiplier (float64), with a shift of 0.
    '''
    
    multiplier: torch.Tensor
    shift     : torch.Tensor
    style     : RequantizeStyle


@dataclass(frozen=True)
class OpSet:
    '''
    #* Hold one version (training, integer, or float) of every op, so that a layer writes its forward pass only once.
    #? mode_ops() returns the set for a mode. The versions of an op take the same arguments.
    '''
    
    requantize         : Callable
    add                : Callable
    pool_avg2d         : Callable
    matmul             : Callable
    normalize_l1       : Callable
    normalize_l1_scores: Callable
    activation         : Callable
    block_quantize     : Callable
