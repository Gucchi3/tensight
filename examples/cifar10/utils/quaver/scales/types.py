# @file types.py
# @brief Types for the scales
# @author Kohei Yamaguchi, Claude

from dataclasses import dataclass
from typing      import Literal

import torch

from ..formats import NumberFormat

GranularityKind = Literal["tensor", "channel", "block"]
LsqInit         = Literal["lsq", "gauss", "mse", "range"]
MathDtype       = Literal["float32", "float64"]



@dataclass(frozen=True)
class Granularity:
    '''
    #* Say how many scales a tensor gets: one in total, one for each channel, or one for each block.
    #? kind is "tensor", "channel" (one for each index of axis), or "block" (block_size values along axis).
    #! One block over the whole axis gives one scale for each token.
    '''
    
    kind      : GranularityKind = "tensor"
    axis      : int             = 0
    block_size: int             = 16


@dataclass(frozen=True)
class BlockScale:
    '''
    #* Hold the block scales of a block-scaled tensor (NVFP4 and the like) as integers, for the exact integer math.
    #? codes: the code of the block scale of each value, in scale_format (real block scale = code x step_min of scale_format). tensor_scale: the second-level scale of the tensor.
    #! The real value of an element = element code x step_min of its format x block scale x tensor_scale.
    '''
    
    codes       : torch.Tensor
    scale_format: NumberFormat
    tensor_scale: torch.Tensor


@dataclass(frozen=True)
class QuantParams:
    '''
    #* Hold the scale, the zero point, and the number format of a quantized tensor.
    #? Real value = (code - zero_point) x code_scale, and code_scale = scale x step_min of the format.
    #! block is set for block-scaled tensors: then scale is block scale x tensor scale for each value, and block keeps its integer parts.
    #! peak is set for the outputs of NEQ: the integer p that the device keeps (scale = p / (2^NORM_BITS x largest value of the format)).
    '''
    
    scale        : torch.Tensor
    zero_point   : torch.Tensor
    number_format: NumberFormat
    block        : BlockScale | None   = None
    peak         : torch.Tensor | None = None
    
    @property
    def code_scale(self) -> torch.Tensor:
        '''
        #* Return the real value of one step of the integer code.
        '''
        
        return self.scale * self.number_format.step_min
