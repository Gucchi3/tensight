# @file quant_block_layer.py
# @brief Shared forward pass of the sample block-scaled conv and linear layers (NVFP4 and the like), with integer math only
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from ..formats    import build_format, ste_attach
from ..ops        import ActivationKind, accumulate_format, accumulate_scale, activation_float, activation_integer, bias_quantize, block_quantize_float, block_quantize_integer, mode_ops, requantize_integer
from ..quantizers import Quantizer, Scaler
from ..scales     import BlockScaler, Granularity, QuantParams, RangeScaler, params_codes, params_values



class QuantBlockLayer(nn.Module):
    '''
    #* Shared forward pass of the sample block-scaled conv and linear layers: a scale for each block of input channels (weights and inputs), an exact wide accumulator, and the output.
    #? QuantBlockConv2d and QuantBlockLinear write the float op (_layer_float) and the exact integer op (_layer_integer). scale_format "ue4m3" gives NVFP4, "uint6" or "uint10" integer scales.
    #! The training mode runs the integer math of the integer mode on the codes (the accumulator is too wide for float values), and takes its gradients from the float layer.
    '''
    
    output_axis: int = 1
    activation : ActivationKind
    
    def __init__(self, layer: nn.Conv2d | nn.Linear, weight_format: str, scale_format: str, input_format: str | None, output_format: str, output_block: bool, activation: ActivationKind, style: str, block_size: int,
                 output_scaler: Scaler | None) -> None:
        '''
        #* Take the weight and the bias of a float layer, and make the quantizers of the weight, of the input blocks, and of the output.
        #? input_format quantizes the input to blocks here (a static tensor scale, and a scale for each block). input_format=None takes the input as it is (block-scaled, or INT8 for W4A8).
        #? output_block=True gives a block-scaled output in output_format (for the next block-scaled layer), and False requantizes it to output_format with one static scale.
        #? output_scaler shares that one scale with other tensors (one SharedScaler for the residual stream of a block: scale sharing). None makes a RangeScaler that follows this output.
        #! Elements round half to even. The input channels must be a multiple of block_size.
        '''
        
        # Check here at the entry: the block scales of a block-scaled output follow each block, so that output cannot share one scale
        if output_block and output_scaler is not None:
            raise ValueError("output_scaler shares one scale of an INT output, but output_block=True gives block scales that follow each block. Use output_block=False.")
        
        # The weight and the bias keep the names of the float layer, so its state_dict loads as it is
        super().__init__()
        self.weight = layer.weight
        self.bias   = layer.bias
        
        # A scale for each block of input channels: axis 1 of the weight, and the channel axis of the activations
        self.weight_blocks    = Granularity("block", axis=1               , block_size=block_size)
        self.blocks           = Granularity("block", axis=self.output_axis, block_size=block_size)
        self.weight_quantizer = Quantizer(build_format(weight_format), BlockScaler(scale_format), self.weight_blocks, "half_even")
        self.input_format     = None if input_format is None else build_format(input_format)
        self.input_scaler     = BlockScaler(scale_format, RangeScaler())
        
        # The output: block-scaled with a static tensor scale (output_block_scaler), or one static scale that follows the maximum or is shared (output_quantizer). Only one of them is used
        self.output_format       = build_format(output_format)
        self.output_block        = output_block
        self.output_block_scaler = BlockScaler(scale_format, RangeScaler())
        self.output_quantizer    = Quantizer(self.output_format, RangeScaler() if output_scaler is None else output_scaler, Granularity(), batch_axis=0)
        
        # Settings of the layer
        self.activation   = activation
        self.style        = style
        self.integer_mode = False
        self.float_mode   = False
        
        return None
    
    def forward(self, x: torch.Tensor, x_params: QuantParams) -> tuple[torch.Tensor, QuantParams]:
        '''
        #* Run the layer: integer codes in the integer mode, fake values in the training mode (the same integer math), and float values in the float mode.
        #? The input is quantized to blocks here when input_format is set. The output is the format of the next op, never the wide accumulator.
        '''
        
        # The input: blocks with a static tensor scale (its scaler follows the input while training), or the input as it is
        ops = mode_ops(self.integer_mode, self.float_mode)
        if self.input_format is not None:
            tensor_scale = self.input_scaler.tensor_scale(x, self.input_format, not self.integer_mode)
            x, x_params  = ops.block_quantize(x, x_params, self.input_format, self.input_scaler.scale_format, tensor_scale, self.blocks, self.style)
        
        # The weight: codes in the integer mode, fake values in the training mode, and the float weight in the float mode
        weight, w_params = (self.weight_quantizer.encode if self.integer_mode else self.weight_quantizer)(self.weight)
        acc_params       = QuantParams(accumulate_scale(x_params, w_params), torch.zeros(1, device=weight.device), accumulate_format(x_params, w_params))
        bias, bias_codes = (None, None) if self.bias is None else bias_quantize(self.bias, x_params, w_params, "half_away")
        view             = [-1 if axis == self.output_axis % x.ndim else 1 for axis in range(x.ndim)]
        
        # Float mode: the float layer, while the scalers of the output see its values
        if self.float_mode:
            acc = activation_float(self._layer_float(x, weight, self.bias), acc_params, self.activation)
            if self.output_block:
                tensor_scale = self.output_block_scaler.tensor_scale(acc, self.output_format, True)
                return block_quantize_float(acc, acc_params, self.output_format, self.output_block_scaler.scale_format, tensor_scale, self.blocks, self.style)
            return acc, self.output_quantizer.measure(acc)
        
        # The exact accumulator from the codes (the training mode takes them back from its fake values), the bias, and the activation on the accumulator
        q_x = x if self.integer_mode else params_codes(x, x_params)
        q_w = weight if self.integer_mode else params_codes(weight, w_params)
        acc = self._layer_integer(q_x, x_params, q_w, w_params)
        acc = acc if bias_codes is None else acc + bias_codes.reshape(view)
        acc = activation_integer(acc, acc_params, self.activation)
        
        # The output codes: blocks with a static tensor scale (its scaler follows the real values while training), or requantized to one static scale
        real = acc.double() * acc_params.scale.double()
        if self.output_block:
            tensor_scale      = self.output_block_scaler.tensor_scale(real, self.output_format, not self.integer_mode)
            codes, out_params = block_quantize_integer(acc, acc_params, self.output_format, self.output_block_scaler.scale_format, tensor_scale, self.blocks, self.style)
            inside            = real.abs() <= self.output_format.max_value * out_params.scale
        else:
            out_params = self.output_quantizer.params(real) if self.integer_mode else self.output_quantizer.measure(real)
            codes      = requantize_integer(acc, acc_params, out_params, self.style)
            grid       = real / out_params.code_scale + out_params.zero_point
            inside     = (grid >= self.output_format.min_value / self.output_format.step_min) & (grid <= self.output_format.max_value / self.output_format.step_min)
        if self.integer_mode:
            return codes, out_params
        
        # Training mode: the values of the integer math in the forward pass, and the gradients of the float layer (with the fake input, weight, and bias) in the backward pass
        reference = activation_float(self._layer_float(x, weight, bias), acc_params, self.activation)
        return ste_attach(reference, params_values(codes, out_params), inside), out_params
    
    def _layer_float(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        '''
        #* Training and float modes: the float op of the layer on fake-quantized (or float) values.
        #? QuantBlockConv2d and QuantBlockLinear write this.
        '''
        
        raise NotImplementedError
    
    def _layer_integer(self, q_x: torch.Tensor, x_params: QuantParams, q_w: torch.Tensor, w_params: QuantParams) -> torch.Tensor:
        '''
        #* Integer mode: the exact op of the layer on integer codes.
        #? QuantBlockConv2d and QuantBlockLinear write this.
        '''
        
        raise NotImplementedError
