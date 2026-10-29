# @file quant_sima_layer.py
# @brief Quantized attention layer: SimA attention and an INT8 residual add on one shared scale, with integer math only (from example/cnn_vit of quaver)
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from utils.quaver                import Granularity, QuantAdd, Quantizer, QuantParams, SharedScaler, build_format
from utils.quaver.ops            import mode_ops
from .noted_quant_sima_attention import NotedQuantSimaAttention



class QuantSimaLayer(nn.Module):
    '''
    #* x + SimA(x) on the INT8 residual stream: the input, the attention output and the sum share one scale.
    #? QuantCifarNet uses one of these. switch_integer_mode() of quaver sets integer_mode.
    '''
    
    def __init__(self, channels: int, heads: int, key_dim: int, value_dim: int) -> None:
        '''
        #* Make the shared scale, the quantizer of the input, the SimA attention (INT8 weights and activations, noting its attention map for tensight), and the add.
        '''
        
        super().__init__()
        self.shared       = SharedScaler(symmetric=True)
        self.residual     = Quantizer(build_format("int8"), self.shared, Granularity(), batch_axis=0)
        self.attention    = NotedQuantSimaAttention(channels, heads, key_dim, value_dim, weight_format="int8", act_format="int8", project_format="int8", output_format="int8", output_scaler=self.shared)
        self.add          = QuantAdd("int8", output_symmetric=True, output_scaler=self.shared)
        self.integer_mode = False
        self.float_mode   = False
        return None
    
    def forward(self, x: torch.Tensor, x_params: QuantParams) -> tuple[torch.Tensor, QuantParams]:
        '''
        #* Return the sum and its parameters: fake values in the training mode, and codes in the integer mode.
        '''
        
        # The input on the shared scale; the scaler forgets the ranges of the last pass
        ops         = mode_ops(self.integer_mode, self.float_mode)
        self.shared.begin_forward()
        params      = self.residual.params(x) if self.integer_mode else self.residual.measure(x)
        x, x_params = ops.requantize(x, x_params, params, "mul16_shift"), params
        
        # SimA attention and the INT8 residual add
        y, y_params = self.attention(x, x_params)
        x, x_params = self.add(y, y_params, x, x_params)
        
        # The shared scale moves toward the widest range of this pass (only while training)
        self.shared.end_forward()
        return x, x_params
