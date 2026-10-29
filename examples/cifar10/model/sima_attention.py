# @file sima_attention.py
# @brief FP32 SimA attention with a BatchNorm after each 1x1 conv (from example/cnn_vit of quaver), noting its attention map for tensight

from torch import nn

import tensight
import torch



NORM_MIN = torch.finfo(torch.float32).eps



class SimaAttention(nn.Module):
    '''
    #* SimA attention: Q and K are L1-normalized instead of a softmax, and (Q K^T / sqrt(key_dim)) V goes through a projection.
    #? SimaLayer uses this. The conv and BatchNorm pairs sit next to each other, so fold_batch_norms() of quaver folds them.
    '''
    
    def __init__(self, channels: int, heads: int, key_dim: int, value_dim: int) -> None:
        '''
        #* Make the 1x1 convs of Q, K, V, and the projection, each with its BatchNorm.
        '''
        
        # Settings
        super().__init__()
        self.heads     = heads
        self.key_dim   = key_dim
        self.value_dim = value_dim
        
        # The 1x1 convs and their BatchNorms
        self.query      = nn.Conv2d(channels         , heads * key_dim  , 1, bias=False)
        self.query_bn   = nn.BatchNorm2d(heads * key_dim)
        self.key        = nn.Conv2d(channels         , heads * key_dim  , 1, bias=False)
        self.key_bn     = nn.BatchNorm2d(heads * key_dim)
        self.value      = nn.Conv2d(channels         , heads * value_dim, 1, bias=False)
        self.value_bn   = nn.BatchNorm2d(heads * value_dim)
        self.project    = nn.Conv2d(heads * value_dim, channels         , 1, bias=False)
        self.project_bn = nn.BatchNorm2d(channels)
        nn.init.zeros_(self.project_bn.weight)
        return None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        #* Return the projected attention output of x (B, channels, height, width).
        '''
        
        # Q, K, and V cut into heads as (B, heads, tokens, dims), and Q and K L1-normalized for each token
        batch, _, height, width = x.shape
        parts                   = ((self.query, self.query_bn), (self.key, self.key_bn), (self.value, self.value_bn))
        query, key, value       = [norm(conv(x)).reshape(batch, self.heads, -1, height * width).transpose(-1, -2) for conv, norm in parts]
        query                   = query / query.abs().sum(dim=-1, keepdim=True).clamp_min(NORM_MIN)
        key                     = key / key.abs().sum(dim=-1, keepdim=True).clamp_min(NORM_MIN)
        
        # The attention map (tensight records it as a query x key map in the recorded steps), times V, and the projection
        scores = tensight.note_tensor("block.attention.map", (query @ key.transpose(-1, -2)) * self.key_dim**-0.5, map="attention_qk")
        output = (scores @ value).transpose(-1, -2).reshape(batch, self.heads * self.value_dim, height, width)
        return self.project_bn(self.project(output))
