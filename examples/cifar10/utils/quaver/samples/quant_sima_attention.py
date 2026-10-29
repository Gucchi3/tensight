# @file quant_sima_attention.py
# @brief Sample quantized SimA attention (L1 normalization instead of softmax), with integer math only
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from ..formats     import RoundMode, build_format
from ..ops         import ActivationKind, NeqScope, OpSet, build_requantize, mode_ops
from ..quantizers  import Quantizer, Scaler
from ..scales      import Granularity, QuantParams, RangeScaler
from .quant_conv2d import QuantConv2d



class QuantSimaAttention(nn.Module):
    '''
    #* SimA attention without softmax: Q and K are L1-normalized for each token and quantized with NEQ (Kaneko and Meng), and then (Q K^T) V is computed, all with integer math on the device.
    #? 1x1 convs for Q, K, V, and the projection, on (B, C, H, W), like the SimA ViT of E_AI_deit (with its 1 / sqrt(key_dim) in the scale).
    #? By default there is no activation before the projection, like the paper. activation="relu6" adds the ReLU6 of E_AI_deit.
    #! NEQ is dynamic (by default one scale for each sample and head, Algorithm 2 of the paper), and the device finds the M0 and n of the attention map with integers.
    '''
    
    activation: ActivationKind
    neq_scope : NeqScope
    round_mode: RoundMode | None
    
    def __init__(self, channels: int, heads: int, key_dim: int, value_dim: int, weight_format: str = "int4", act_format: str = "int4", project_format: str = "int4", activation: ActivationKind = "none", neq_scope: NeqScope = "tensor",
                 output_format: str | None = None, output_scaler: Scaler | None = None, style: str = "mul16_shift", round_mode: RoundMode | None = None) -> None:
        '''
        #* Make the four 1x1 convs, and the quantizers of the attention map and of the projection input.
        #? act_format is for Q, K, the attention map, and V. activation ("none", "relu", or "relu6") runs on attention x V. neq_scope: "tensor" (the paper's setting) or "token".
        #? output_format requantizes the projection to it (fused into the kernel on the device), with output_scaler when the scale is shared with the residual (scale sharing). None returns the INT32 accumulator.
        #! Use a signed project_format without an activation, and an unsigned one after ReLU or ReLU6 (an unsigned format cuts the negative values, like a ReLU).
        #! style needs a multiplier and a shift (mul16_shift or shift_only), and act_format needs codes of 8 bits or less, so the integer math stays exact.
        '''
        
        # Check here at the entry: the dynamic scales need M0 and n, and the products must stay exact
        number_format = build_format(act_format)
        if build_requantize(style).multiplier_max == 0:
            raise ValueError(f"SimA needs a style with a multiplier and a shift (mul16_shift or shift_only), not {style!r}: its scales change while running.")
        if number_format.max_value / number_format.step_min > 255:
            raise ValueError(f"act_format must have codes of 8 bits or less (INT8, INT4, FP4, ...), not {act_format!r}.")
        
        # Settings
        super().__init__()
        self.heads         = heads
        self.key_dim       = key_dim
        self.value_dim     = value_dim
        self.weight_format = weight_format
        self.act_format    = number_format
        self.activation    = activation
        self.neq_scope     = neq_scope
        self.style         = style
        self.round_mode    = round_mode
        self.integer_mode  = False
        self.float_mode    = False
        
        # 1x1 convs for Q, K, V, and the projection
        self.query   = self._sima_conv(channels         , heads * key_dim  , None         , None         )
        self.key     = self._sima_conv(channels         , heads * key_dim  , None         , None         )
        self.value   = self._sima_conv(channels         , heads * value_dim, act_format   , None         )
        self.project = self._sima_conv(heads * value_dim, channels         , output_format, output_scaler)
        
        # The scales of the attention map and of the projection input follow the maximum (axis 0 is the batch)
        self.attention_quantizer = Quantizer(self.act_format             , RangeScaler(), Granularity(), round_mode, batch_axis=0)
        self.project_quantizer   = Quantizer(build_format(project_format), RangeScaler(), Granularity(), round_mode, batch_axis=0)
        
        return None
    
    def forward(self, x: torch.Tensor, x_params: QuantParams) -> tuple[torch.Tensor, QuantParams]:
        '''
        #* Run SimA attention in the current mode (fake values, integer codes, or float values).
        #? Returns the output of the projection and its parameters, ready for a residual add: the INT32 accumulator, or output_format codes.
        #! x_params must have one scale for the whole tensor.
        '''
        
        # Q and K: 1x1 conv to the INT32 accumulator, cut into heads, and NEQ (one scale for each sample and head, or for each token)
        ops                     = mode_ops(self.integer_mode, self.float_mode)
        batch, _, height, width = x.shape
        query, query_params     = self._sima_tokens(self.query, x, x_params, ops)
        key, key_params         = self._sima_tokens(self.key  , x, x_params, ops)
        
        # Attention map: exact Q K^T, requantized with the integer maxima of NEQ, and 1 / sqrt(key_dim) in the static part of the multiplier
        factor               = self.key_dim**-0.5
        key_t_params         = QuantParams(key_params.scale.transpose(-1, -2), key_params.zero_point.transpose(-1, -2), key_params.number_format, peak=None if key_params.peak is None else key_params.peak.transpose(-1, -2))
        scores, score_params = ops.matmul(query, query_params, key.transpose(-1, -2), key_t_params)
        attention_params     = self.attention_quantizer.params(scores) if self.integer_mode else self.attention_quantizer.measure(scores * factor)
        attention            = ops.normalize_l1_scores(scores, score_params, query_params, key_t_params, attention_params, factor, self.style)
        
        # Values: 1x1 conv to the activation format, cut into heads
        value, value_params = self.value(x, x_params)
        value               = value.reshape(batch, self.heads, self.value_dim, -1).transpose(-1, -2)
        
        # Output: exact attention x V, the activation on the accumulator (none by default), then requantize to the projection input (static M0 and n)
        output, output_params = ops.matmul(attention, attention_params, value, value_params)
        output                = ops.activation(output, output_params, self.activation)
        project_params        = self.project_quantizer.params(output) if self.integer_mode else self.project_quantizer.measure(output)
        output                = ops.requantize(output, output_params, project_params, self.style)
        output                = output.transpose(-1, -2).reshape(batch, self.heads * self.value_dim, height, width)
        return self.project(output, project_params)
    
    def _sima_conv(self, in_channels: int, out_channels: int, output_format: str | None, output_scaler: Scaler | None) -> QuantConv2d:
        '''
        #* Make one 1x1 conv with the shared weight format, style, and rounding.
        #? __init__() uses this for the four convs.
        '''
        
        return QuantConv2d(in_channels, out_channels, 1, weight_format=self.weight_format, output_format=output_format, style=self.style, round_mode=self.round_mode, output_scaler=output_scaler)
    
    def _sima_tokens(self, conv: QuantConv2d, x: torch.Tensor, x_params: QuantParams, ops: OpSet) -> tuple[torch.Tensor, QuantParams]:
        '''
        #* Run the Q or K conv to its accumulator, cut it into heads as (B, heads, tokens, key_dim), and run NEQ on it.
        #? forward() uses this for Q and K. The accumulator has one scale for each channel, so the scales are cut into heads in the same way.
        '''
        
        acc, acc_params  = conv(x, x_params)
        acc, scale, zero = [part.reshape(part.shape[0], self.heads, self.key_dim, -1).transpose(-1, -2) for part in (acc, acc_params.scale, acc_params.zero_point)]
        return ops.normalize_l1(acc, QuantParams(scale, zero, acc_params.number_format), self.act_format, self.style, self.neq_scope)
