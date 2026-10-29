# @file quantizer.py
# @brief Quantizer: number format + scaler + granularity + rounding
# @author Kohei Yamaguchi, Claude

from torch import nn

import math
import torch

from ..formats import NumberFormat, RoundMode, format_round_mode, ste_quantize
from ..scales  import BlockScaler, Granularity, QuantParams, compute_pow2, group_count, params_note
from .lsq      import lsq_grad_scale, lsq_quantize, lsq_scale_gradient
from .types    import Scaler



class Quantizer(nn.Module):
    '''
    #* Quantize tensors with a number format, a scaler, a granularity, and a rounding mode.
    #? forward() is the training mode (fake quantization), and encode() is the integer mode (codes).
    #! With enabled = False, forward() only updates the scaler and returns x as it is.
    '''
    
    round_mode: RoundMode
    
    def __init__(self, number_format: NumberFormat, scaler: Scaler, granularity: Granularity = Granularity(), round_mode: RoundMode | None = None, pow2: bool = False, batch_axis: int | None = None) -> None:
        '''
        #* Keep the parts of the quantizer.
        #? round_mode=None uses the usual rounding of the format (half_even for FP, half_away for INT). pow2=True rounds the scale up to a power of two (for devices that only shift).
        #! Give batch_axis (the axis of the samples, usually 0) to activation quantizers: the LSQ gradient scale then counts the values of one sample, as in the LSQ paper. Weights keep None.
        '''
        
        # Check here at the entry: FP formats have no zero point, so their scalers must be symmetric, and block scales are on their own grid, not powers of two
        if number_format.kind == "fp" and not getattr(scaler, "symmetric", True):
            raise ValueError(f"Asymmetric quantization works only for INT formats, not for {number_format.name}. Use a symmetric scaler.")
        if pow2 and isinstance(scaler, BlockScaler):
            raise ValueError("pow2 does not work with BlockScaler: its block scales are on the grid of their own format.")
        
        # Keep the parts
        super().__init__()
        self.number_format = number_format
        self.scaler        = scaler
        self.granularity   = granularity
        self.round_mode    = format_round_mode(number_format, round_mode)
        self.pow2          = pow2
        self.batch_axis    = batch_axis
        self.enabled       = True
        
        return None
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, QuantParams]:
        '''
        #* Training mode: return the fake-quantized x and its parameters.
        #! It computes and returns float32.
        '''
        
        # Get the scale and the zero point from the scaler
        x      = x.float()
        params = self._params(x, update=True)
        if not self.enabled:
            return x, params
        
        # A learned scale gets the LSQ gradient, and the others get a straight-through gradient
        if self.scaler.learnable:
            return lsq_quantize(x, params.scale, params.zero_point, self.number_format, self.round_mode, self._grad_scale(x)), params
        return ste_quantize(x, params.scale, params.zero_point, self.number_format, self.round_mode), params
    
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, QuantParams]:
        '''
        #* Integer mode: return the integer codes (int64) of x and its parameters.
        #? Real value = (code - zero_point) x params.code_scale. The codes come from the device op torch.ops.quaver.quantize, so torch.export keeps it as one node.
        #! It does not change the scaler.
        '''
        
        with torch.no_grad():
            params = self._params(x.float(), update=False)
            codes  = torch.ops.quaver.quantize(x.float(), params.scale, params.zero_point, self.number_format.name, self.round_mode)
        return params_note(codes, params), params
    
    def params(self, x: torch.Tensor) -> QuantParams:
        '''
        #* Return the current parameters for a tensor with the shape of x, without changing the scaler.
        #? x only gives the shape, except for scalers that measure every time (dynamic, nvfp4).
        '''
        
        with torch.no_grad():
            return self._params(x.float(), update=False)
    
    def measure(self, x: torch.Tensor) -> QuantParams:
        '''
        #* Let the scaler see x (it updates while training), and return the parameters without quantizing.
        #? Layers use this for outputs that the integer ops round, not this quantizer. A learned step comes with the LSQ gradient scale, so those ops can train it.
        '''
        
        params = self._params(x.detach().float(), update=True)
        if not self.scaler.learnable:
            return params
        return QuantParams(lsq_scale_gradient(params.scale, self._grad_scale(x)), params.zero_point, params.number_format)
    
    def _params(self, x: torch.Tensor, update: bool) -> QuantParams:
        '''
        #* Ask the scaler for the scale and the zero point, and round the scale to a power of two if pow2 is on.
        #? forward(), encode(), params(), and measure() use this. A BlockScaler also gives the integer parts of its block scales.
        '''
        
        # Block scales come with their integer parts, for the exact integer math
        if isinstance(self.scaler, BlockScaler):
            scale, zero_point, block = self.scaler.split(x, self.number_format, self.granularity, update)
            return QuantParams(scale, zero_point, self.number_format, block)
        
        # Other scalers give the scale and the zero point
        scale, zero_point = self.scaler(x, self.number_format, self.granularity, update)
        if self.pow2:
            scale = scale + (compute_pow2(scale.detach()) - scale).detach()
        return QuantParams(scale, zero_point, self.number_format)
    
    def _grad_scale(self, x: torch.Tensor) -> float:
        '''
        #* Return the LSQ gradient scale for x: 1 / sqrt(N x Qp), where N is the number of values that share one scale.
        #? forward() and measure() use this. With batch_axis, N is counted in one sample, so the gradient does not change with the batch size.
        '''
        
        shape = list(x.shape)
        if self.batch_axis is not None:
            shape[self.batch_axis] = 1
        return lsq_grad_scale(math.prod(shape) // group_count(torch.Size(shape), self.granularity), self.number_format)
