# @file block_scaler.py
# @brief The two levels of scales of block-scaled formats (NVFP4, integer block scales): one for each block, and one for the tensor
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from ..formats       import NumberFormat, build_format, format_to_integer, integer_format
from .compute        import SCALE_MIN, compute_block
from .dynamic_scaler import DynamicScaler
from .group          import group_merge, group_split
from .types          import BlockScale, Granularity

# A format with the largest value 1: a scaler asked for a scale in it returns the range itself
UNIT_FORMAT = integer_format(1, signed=False)



class BlockScaler(nn.Module):
    '''
    #* Make the two scales of a block-scaled format: one for each block (rounded to scale_format), and one for the tensor.
    #? scale_format "ue4m3" (unsigned E4M3, up to 448) gives NVFP4, and "uint6" or "uint10" gives integer block scales (VS-Quant). Use it with Granularity("block", axis, 16).
    #? tensor_scaler measures the range of the tensor: DynamicScaler (the default) every time, for weights. A RangeScaler keeps the tensor scale static, for activations.
    #! The block scales always follow each block. The zero point is always 0.
    '''
    
    learnable: bool = False
    
    def __init__(self, scale_format: str = "ue4m3", tensor_scaler: nn.Module | None = None) -> None:
        '''
        #* Keep the format of the block scales and the scaler of the tensor.
        '''
        
        super().__init__()
        self.scale_format  = build_format(scale_format)
        self.tensor_scaler = DynamicScaler() if tensor_scaler is None else tensor_scaler
        
        return None
    
    def forward(self, x: torch.Tensor, number_format: NumberFormat, granularity: Granularity, update: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        #* Return the scale of each value (block scale x tensor scale) and a zero point of 0.
        '''
        
        scale, zero_point, _ = self.split(x, number_format, granularity, update)
        return scale, zero_point
    
    def split(self, x: torch.Tensor, number_format: NumberFormat, granularity: Granularity, update: bool) -> tuple[torch.Tensor, torch.Tensor, BlockScale]:
        '''
        #* Return the scale of each value, the zero point 0, and the integer parts of the block scales (the code of each value's block scale, and the tensor scale).
        #? Quantizer uses this, so that the QuantParams of block-scaled tensors carry their block scales.
        '''
        
        # The tensor scale, and the scale of each block on the grid of scale_format
        tensor_scale = self.tensor_scale(x, number_format, update)
        block_scale  = compute_block(group_split(x.detach(), granularity).abs().amax(dim=1), tensor_scale, number_format, self.scale_format)
        
        # Spread the scales over the values, and keep the codes of the block scales
        scale = group_merge(block_scale * tensor_scale                       , x.shape, granularity)
        codes = group_merge(format_to_integer(block_scale, self.scale_format), x.shape, granularity)
        return scale, torch.zeros_like(scale), BlockScale(codes, self.scale_format, tensor_scale)
    
    def tensor_scale(self, x: torch.Tensor, number_format: NumberFormat, update: bool) -> torch.Tensor:
        '''
        #* Return the tensor scale: the largest |value| from the tensor scaler / (top of number_format x the largest block scale), so the largest block gets the top block scale.
        #? Layers that quantize activations to blocks with the integer math use this for the static tensor scale (a RangeScaler as the tensor scaler).
        '''
        
        tensor_amax = self.tensor_scaler(x, UNIT_FORMAT, Granularity(), update)[0]
        return (tensor_amax / (self.scale_format.max_value * number_format.max_value)).clamp_min(SCALE_MIN)
