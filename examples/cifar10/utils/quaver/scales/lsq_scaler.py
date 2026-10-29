# @file lsq_scaler.py
# @brief Learned scale (LSQ) and learned zero point (LSQ+)
# @author Kohei Yamaguchi, Claude

from torch  import nn
from typing import Any

import math
import torch

from ..formats import NumberFormat, ste_round
from .compute  import compute_mse_range, compute_scale
from .group    import group_merge, group_split
from .minmax   import minmax_measure
from .types    import Granularity, LsqInit

STEP_MIN = torch.finfo(torch.float32).eps



class LsqScaler(nn.Module):
    '''
    #* Learn the scale (the step) with gradient descent (LSQ). Asymmetric also learns the zero point (LSQ+).
    #? Quantizer gives the LSQ gradient with lsq_quantize(), and the fake ops give it with params_attach(). The first training batch sets the start values.
    #! Call lsq_clamp_steps() after each optimizer step to keep the step above 0.
    '''
    
    learnable: bool = True
    
    def __init__(self, scale_count: int = 1, symmetric: bool = True, init: LsqInit = "mse") -> None:
        '''
        #* Make the step (NaN until the first batch), and the zero point for asymmetric.
        #? init: "mse" (least squared error, LSQ+ for activations), "gauss" (mean +- 3 std, LSQ+ for weights), "lsq" (2 x mean|x| / sqrt(Qp), LSQ), or "range" (min and max).
        #! "lsq" is only for symmetric steps. Asymmetric with "lsq" starts from the measured range.
        '''
        
        super().__init__()
        self.symmetric   = symmetric
        self.init        = init
        self.initialized = False
        self.step        = nn.Parameter(torch.full((scale_count,), torch.nan))
        self.offset      = None if symmetric else nn.Parameter(torch.zeros(scale_count))
        
        return None
    
    def forward(self, x: torch.Tensor, number_format: NumberFormat, granularity: Granularity, update: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        #* Return the learned step and zero point (with gradients). Both broadcast against x.
        #! The zero point is rounded to an integer with a straight-through gradient.
        '''
        
        # Set the start values on the first training batch
        if update and self.training and not self.initialized:
            with torch.no_grad():
                # The range: measured, the least squared error (LSQ+ for activations), or mean +- 3 std (LSQ+ for weights)
                groups    = group_split(x.detach(), granularity)
                low, high = minmax_measure(x, granularity, None, self.symmetric)
                if self.init == "mse":
                    low, high = compute_mse_range(groups, low, high, number_format, self.symmetric)
                elif self.init == "gauss":
                    center, spread = groups.mean(dim=1), 3 * groups.std(dim=1, correction=0)
                    low, high      = center - spread, center + spread
                
                # Turn the range into the step and the zero point. Symmetric steps of the papers: LSQ uses 2 x mean|x| / sqrt(Qp), and LSQ+ puts the larger end on the end code (2^(b-1) for signed INT)
                scale, zero_point = compute_scale(low, high, number_format, self.symmetric)
                if self.symmetric and self.init == "lsq":
                    size  = groups.abs() if number_format.signed else groups.clamp_min(0.0)
                    scale = (2.0 * size.mean(dim=1) / math.sqrt(number_format.max_value)).clamp_min(STEP_MIN)
                elif self.symmetric and self.init == "gauss":
                    scale = (torch.maximum(low.abs(), high.abs()) / max(-number_format.min_value, number_format.max_value)).clamp_min(STEP_MIN)
                
                # Keep them
                self.step.copy_(scale)
                if self.offset is not None:
                    self.offset.copy_(zero_point)
            self.initialized = True
        
        # Keep the step above 0, and the zero point on an integer code
        step = self.step.clamp_min(STEP_MIN)
        if self.offset is None:
            zero_point = torch.zeros_like(step)
        else:
            zero_point = ste_round(self.offset, "half_even").clamp(number_format.min_value, number_format.max_value)
        return group_merge(step, x.shape, granularity), group_merge(zero_point, x.shape, granularity)
    
    def _load_from_state_dict(self, state_dict: dict[str, Any], prefix: str, local_metadata: dict[str, Any], strict: bool, missing_keys: list[str], unexpected_keys: list[str], error_msgs: list[str]) -> None:
        '''
        #* After a saved step is loaded, mark the scaler as started.
        #? PyTorch calls this from load_state_dict().
        #! This reads a value from the GPU. That is fine here, because loading is not part of a training step.
        '''
        
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        self.initialized = bool(torch.isfinite(self.step).all())
        
        return None
