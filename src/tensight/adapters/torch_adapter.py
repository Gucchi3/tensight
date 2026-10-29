# @file torch_adapter.py
# @brief TorchAdapter: reads PyTorch quantized tensors (C-04)

from torch  import nn
from typing import Any

import torch

from ..types import QuantInfo, QuantizerSite, QuantPair, QuantValue



TORCH_FORMATS = {
    torch.qint8   : (8 , -128     ),
    torch.quint8  : (8 , 0        ),
    torch.qint32  : (32, -(2**31) ),
    torch.quint4x2: (4 , 0        ),
    torch.quint2x4: (2 , 0        ),
}
PER_TENSOR    = (torch.per_tensor_affine, torch.per_tensor_symmetric)



class TorchAdapter:
    '''
    #* Read the quantization info of PyTorch quantized tensors (qint8, quint8, and the others), and the calls of the Quantize layers (FR-012 to FR-015).
    #? registry_adapters() tries this one last.
    '''
    
    name = "torch"
    
    def find_quantizers(self, model: nn.Module) -> list[QuantizerSite]:
        '''
        #* Find the Quantize layers, which turn float tensors into quantized tensors (03 U-029, BUG-005).
        '''
        
        return [QuantizerSite(module, "output", name, None, "forward") for name, module in model.named_modules() if isinstance(module, torch.ao.nn.quantized.Quantize)]
    
    def read_value(self, value: object) -> QuantValue | None:
        '''
        #* Read a quantized tensor: its real values, codes, scales, zero points, and levels.
        '''
        
        # Handle only quantized tensors
        if not isinstance(value, torch.Tensor) or not value.is_quantized:
            return None
        bits, low = TORCH_FORMATS[value.dtype]
        
        # Shape the scales and the zero points so that they broadcast to the values
        if value.qscheme() in PER_TENSOR:
            shape      = (1,) * value.ndim
            scale      = torch.full(shape, value.q_scale(), dtype=torch.float64)
            zero_point = torch.full(shape, float(value.q_zero_point()), dtype=torch.float64)
            per        = "tensor"
        else:
            shape       = [1] * value.ndim
            axis        = value.q_per_channel_axis()
            shape[axis] = -1
            scale       = value.q_per_channel_scales().to(torch.float64).reshape(shape)
            zero_point  = value.q_per_channel_zero_points().to(torch.float64).reshape(shape)
            per         = "channel"
        
        # Make the info; the levels are the codes, without the zero point (02a 5章)
        total  = 2**bits
        levels = torch.arange(low, low + total, dtype=torch.float64) if total <= 4096 else None
        info   = QuantInfo(str(value.dtype).removeprefix("torch."), "int", bits, scale, zero_point, per, value.int_repr().to(torch.int64), total, levels, low, "nearest")
        return QuantValue(value.dequantize(), info)
    
    def read_quantizer(self, site: QuantizerSite, args: tuple[Any, ...], result: object) -> QuantPair | None:
        '''
        #* Read one call of a Quantize layer: the float input, and the real values and the info of the quantized output (03 U-029, BUG-005).
        '''
        
        # Handle only a float tensor turned into a quantized tensor
        before = args[0] if args else None
        found  = self.read_value(result)
        if not isinstance(before, torch.Tensor) or not before.is_floating_point() or found is None:
            return None
        return QuantPair(before, found.values, found.info)
