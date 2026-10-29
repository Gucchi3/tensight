# @file types.py
# @brief Types for the quantizers
# @author Kohei Yamaguchi, Claude

from dataclasses import dataclass
from typing      import Protocol

import torch

from ..formats import NumberFormat, RoundMode
from ..scales  import Granularity, GranularityKind, LsqInit



@dataclass(frozen=True)
class QuantizerSpec:
    '''
    #* Hold the settings to build a Quantizer from a config file.
    #? build_quantizer() turns this into a Quantizer. Copy the values of your config into it. round_mode=None uses the usual rounding of the format (half_even for FP, half_away for INT).
    #! Set batch_axis (usually 0) for activations, so that the LSQ gradient scale counts one sample. freeze_step stops the max, percentile, and shared ranges after that many updates.
    '''
    
    format     : str              = "int8"
    granularity: GranularityKind  = "tensor"
    axis       : int              = 0
    block_size : int              = 16
    scaler     : str              = "max"
    symmetric  : bool             = True
    round_mode : RoundMode | None = None
    momentum   : float            = 0.95
    percentile : float | None     = 99.9
    pow2       : bool             = False
    lsq_init   : LsqInit          = "mse"
    batch_axis : int | None       = None
    freeze_step: int | None       = None


class Scaler(Protocol):
    '''
    #* Describe what a scaler has: learnable, and a call that returns the scale and the zero point.
    #? Write your own scaler in this shape to pass it to Quantizer.
    '''
    
    learnable: bool
    
    def __call__(self, x: torch.Tensor, number_format: NumberFormat, granularity: Granularity, update: bool) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        #* Return the scale and the zero point. Both broadcast against x.
        #! update=False must not change any state.
        '''
        
        ...
