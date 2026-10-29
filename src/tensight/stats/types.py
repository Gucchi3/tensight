# @file types.py
# @brief Data containers of the statistics (C-03)

from dataclasses import dataclass

import torch

from ..types import MapPattern, PointInfo, QuantInfo, QuantPair



@dataclass(frozen=True)
class MapPlan:
    '''
    #* Hold a map pattern fitted to the shape of a tensor (03 U-021).
    '''
    
    reduce   : tuple[int, ...]
    rows     : tuple[int, ...]
    cols     : tuple[int, ...]
    row_size : int
    col_size : int
    row_names: tuple[str, ...]
    col_names: tuple[str, ...]


@dataclass(eq=False)
class PendingReport:
    '''
    #* Hold the summary of one tensor on its device, before it moves to the CPU (03 U-023).
    #! numbers holds low, high, total, spread, count, zero, nan, inf, and square, in this order.
    '''
    
    dtype        : str
    shape        : tuple[int, ...]
    device       : str
    numel        : int
    bins         : int
    numbers      : torch.Tensor
    counts       : torch.Tensor
    sample       : torch.Tensor
    elements     : torch.Tensor | None
    element_index: tuple[int, ...]
    map          : torch.Tensor | None
    plan         : MapPlan | None
    pattern      : MapPattern | None
    compare      : torch.Tensor | None
    full         : torch.Tensor | None


@dataclass(eq=False)
class PendingQuant:
    '''
    #* Hold the summary of the quantization of one tensor on its device (03 U-025).
    #! scale and zero_point hold count, min, max, mean. clip holds outside and finite. errors holds sum e^2, max |e|, sum before^2, and count.
    '''
    
    format     : str
    kind       : str
    bits       : int
    granularity: str
    code_total : int
    code_min   : int
    round_mode : str
    scale      : torch.Tensor
    zero_point : torch.Tensor | None
    counts     : torch.Tensor | None
    used       : torch.Tensor | None
    levels     : torch.Tensor | None
    clip       : torch.Tensor | None
    errors     : torch.Tensor | None


@dataclass(eq=False)
class CollectorEntry:
    '''
    #* Hold everything that StepCollector gathers for one point in one step (03 U-027).
    '''
    
    point    : PointInfo
    pending  : PendingReport | None = None
    quant    : PendingQuant | None  = None
    wait     : QuantPair | None     = None
    pair_info: QuantInfo | None     = None
    boundary : torch.Tensor | None  = None
    ratio    : torch.Tensor | None  = None
    scales   : torch.Tensor | None  = None
    zeros    : torch.Tensor | None  = None
