# @file quant_cifar_net.py
# @brief Quantized CifarNet made from the quaver samples (INT8 weights and activations), with integer math only

from torch import nn

import torch

from utils.quaver      import QuantConv2d, QuantLinear, QuantParams, build_format
from .quant_sima_layer import QuantSimaLayer



class QuantCifarNet(nn.Module):
    '''
    #* The QAT version of CifarNet: the same layer names, so the folded FP32 weights load into it.
    #? main.py fine-tunes it with QAT. In the integer mode it takes uint8 codes and returns (INT32 codes, parameters).
    '''
    
    input_scale: torch.Tensor
    
    def __init__(self) -> None:
        '''
        #* Make conv1 (ReLU into uint8), conv2 (into the INT32 accumulator for the shared scale), the attention layer, and the classifier (INT32 logits).
        '''
        
        super().__init__()
        self.conv1        = QuantConv2d(3 , 32, 3, 2, 1, weight_format="int8", output_format="uint8", activation="relu")
        self.conv2        = QuantConv2d(32, 64, 3, 2, 1, weight_format="int8", output_format=None   , activation="none")
        self.block        = QuantSimaLayer(64, 4, 16, 16)
        self.classifier   = QuantLinear(64 * 8 * 8, 10, weight_format="int8")
        self.integer_mode = False
        self.register_buffer("input_scale", torch.tensor(1 / 255))
        return None
    
    def forward(self, images: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, QuantParams]:
        '''
        #* Return the logits: fake values in the training mode, and (codes, parameters) in the integer mode.
        '''
        
        # The input is the uint8 code with the scale 1/255
        params = QuantParams(self.input_scale, torch.zeros_like(self.input_scale), build_format("uint8"))
        x      = images.long() if self.integer_mode else images.float() * self.input_scale
        
        # The layers
        x, params = self.conv1(x, params)
        x, params = self.conv2(x, params)
        x, params = self.block(x, params)
        x, params = self.classifier(x.flatten(1), params)
        return (x, params) if self.integer_mode else x
