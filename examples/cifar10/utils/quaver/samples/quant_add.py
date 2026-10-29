# @file quant_add.py
# @brief Sample quantized residual add
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from ..formats    import RoundMode, build_format
from ..ops        import mode_ops
from ..quantizers import Quantizer, Scaler
from ..scales     import Granularity, QuantParams, RangeScaler



class QuantAdd(nn.Module):
    '''
    #* Residual add: rescale both inputs to the output scale, add them, round the sum once, and clip, like the device.
    #? forward(a, a_params, b, b_params) returns (output, output params). The output scale follows the range of the sum, or is shared (output_scaler).
    #? Scale sharing (Kaneko and Meng): when a, b, and the output use one SharedScaler, the ratios are 1 and the add is a plain integer add with saturation (M0 = 2^15, n = 15 is exact).
    '''
    
    def __init__(self, output_format: str = "int8", momentum: float = 0.95, style: str = "mul16_shift", round_mode: RoundMode | None = None, output_symmetric: bool = True, output_scaler: Scaler | None = None) -> None:
        '''
        #* Make the output quantizer, and keep the settings.
        #? output_scaler gives the output a shared scale (a SharedScaler). None makes a RangeScaler that follows the sum.
        #! output_symmetric=False gives the output a zero point (INT formats only).
        '''
        
        super().__init__()
        self.output_quantizer = Quantizer(build_format(output_format), RangeScaler(momentum=momentum, symmetric=output_symmetric) if output_scaler is None else output_scaler, Granularity(), round_mode, batch_axis=0)
        self.style            = style
        self.integer_mode     = False
        self.float_mode       = False
        
        return None
    
    def forward(self, a: torch.Tensor, a_params: QuantParams, b: torch.Tensor, b_params: QuantParams) -> tuple[torch.Tensor, QuantParams]:
        '''
        #* Add a and b in the current mode (fake values, integer codes, or float values).
        '''
        
        ops        = mode_ops(self.integer_mode, self.float_mode)
        out_params = self.output_quantizer.params(a) if self.integer_mode else self.output_quantizer.measure(a + b)
        return ops.add(a, a_params, b, b_params, out_params, self.style), out_params
