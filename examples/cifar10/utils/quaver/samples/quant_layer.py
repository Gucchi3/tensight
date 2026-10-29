# @file quant_layer.py
# @brief Shared forward pass of the sample conv and linear layers
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from ..formats    import RoundMode, build_format, format_round_mode
from ..ops        import ACCUMULATOR_FORMAT, ActivationKind, accumulate_scale, bias_quantize, mode_ops
from ..quantizers import Quantizer, Scaler
from ..scales     import DynamicScaler, Granularity, QuantParams, RangeScaler



class QuantLayer(nn.Module):
    '''
    #* Shared forward pass of the sample conv and linear layers: weight, INT32 bias, accumulator, activation, and requantization.
    #? QuantConv2d and QuantLinear write the float op (_layer_float) and the exact integer op (_layer_integer).
    #! Set integer_mode = True (or call switch_integer_mode()) to take and return integer codes, like the device. switch_quantizers(model, False) sets float_mode = True (calibration).
    '''
    
    output_axis: int = 1
    activation : ActivationKind
    round_mode : RoundMode | None
    
    def __init__(self, layer: nn.Conv2d | nn.Linear, weight_format: str, output_format: str | None, activation: ActivationKind, style: str, round_mode: RoundMode | None, weight_symmetric: bool, output_symmetric: bool,
                 output_scaler: Scaler | None) -> None:
        '''
        #* Take the weight and the bias of a float layer, and make the quantizers of the weight and the output.
        #? The float layer gives the usual start values of PyTorch. output_format=None returns the INT32 accumulator without requantizing. round_mode=None uses the usual rounding of each format.
        #? output_scaler gives the output a scale shared with other tensors (one SharedScaler for the residual stream of a block: scale sharing). None makes a RangeScaler that follows this output.
        #! weight_symmetric and output_symmetric choose symmetric scales (zero point 0) or asymmetric ones (with a zero point, for INT formats only).
        '''
        
        # The weight and the bias keep the names of the float layer, so its state_dict loads as it is
        super().__init__()
        self.weight = layer.weight
        self.bias   = layer.bias
        
        # One weight scale for each output channel, measured every time, and one output scale that follows the range or is shared (axis 0 of the output is the batch)
        self.weight_quantizer = Quantizer(build_format(weight_format), DynamicScaler(symmetric=weight_symmetric), Granularity("channel", axis=0), round_mode)
        self.output_quantizer = None if output_format is None else Quantizer(build_format(output_format), RangeScaler(symmetric=output_symmetric) if output_scaler is None else output_scaler, Granularity(), round_mode, batch_axis=0)
        
        # Settings of the layer
        self.activation   = activation
        self.style        = style
        self.round_mode   = round_mode
        self.integer_mode = False
        self.float_mode   = False
        
        return None
    
    def forward(self, x: torch.Tensor, x_params: QuantParams) -> tuple[torch.Tensor, QuantParams]:
        '''
        #* Run the layer: on fake-quantized values in the training mode, on integer codes in the integer mode, and on float values in the float mode.
        #? After the accumulator, the training and integer modes run the same integer math, so their outputs are the same. The float mode uses the float weight and bias.
        '''
        
        # Quantize the weight: fake values in the training mode, codes in the integer mode, and the float weight when the quantizer is off
        ops              = mode_ops(self.integer_mode, self.float_mode)
        quantize         = self.weight_quantizer.encode if self.integer_mode else self.weight_quantizer
        weight, w_params = quantize(self.weight)
        
        # Accumulate with the float op or the exact integer op, and add the INT32 bias (the float mode adds the float bias)
        if self.bias is None:
            acc = self._layer_integer(x, x_params, weight, w_params) if self.integer_mode else self._layer_float(x, weight, None)
        else:
            bias, bias_codes = bias_quantize(self.bias, x_params, w_params, format_round_mode(ACCUMULATOR_FORMAT, self.round_mode))
            bias             = self.bias if self.float_mode else bias
            acc              = self._layer_integer(x, x_params, weight, w_params) + self._layer_view(bias_codes, x.ndim) if self.integer_mode else self._layer_float(x, weight, bias)
        
        # The accumulator has one scale for each output channel. Apply the activation on it
        scale      = self._layer_view(accumulate_scale(x_params, w_params), acc.ndim)
        acc_params = QuantParams(scale, torch.zeros_like(scale), ACCUMULATOR_FORMAT)
        acc        = ops.activation(acc, acc_params, self.activation)
        if self.output_quantizer is None:
            return acc, acc_params
        
        # Requantize to the output format, like the device
        out_params = self.output_quantizer.params(acc) if self.integer_mode else self.output_quantizer.measure(acc)
        return ops.requantize(acc, acc_params, out_params, self.style), out_params
    
    def _layer_float(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        '''
        #* Training and float modes: the float op of the layer on fake-quantized (or float) values.
        #? QuantConv2d and QuantLinear write this.
        '''
        
        raise NotImplementedError
    
    def _layer_integer(self, q_x: torch.Tensor, x_params: QuantParams, q_w: torch.Tensor, w_params: QuantParams) -> torch.Tensor:
        '''
        #* Integer mode: the exact op of the layer on integer codes.
        #? QuantConv2d and QuantLinear write this.
        '''
        
        raise NotImplementedError
    
    def _layer_view(self, values: torch.Tensor, ndim: int) -> torch.Tensor:
        '''
        #* Reshape one value for each output channel, so that it broadcasts against the output.
        #? forward() uses this for the bias and the scale.
        '''
        
        view                   = [1] * ndim
        view[self.output_axis] = -1
        return values.reshape(view)
