# @file sima_layer.py
# @brief FP32 attention layer: SimA attention with a residual add
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from .sima_attention import SimaAttention



class SimaLayer(nn.Module):
    '''
    #* One attention layer on the tokens: x + SimA(x).
    #? CifarNet uses one of these. The name "attention" matches QuantSimaLayer, so the folded weights load into the QAT model.
    '''
    
    def __init__(self, channels: int, heads: int, key_dim: int, value_dim: int) -> None:
        '''
        #* Make the attention.
        '''
        
        super().__init__()
        self.attention = SimaAttention(channels, heads, key_dim, value_dim)
        return None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        #* Add the attention output to x.
        '''
        
        return x + self.attention(x)
