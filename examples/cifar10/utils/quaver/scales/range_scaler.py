# @file range_scaler.py
# @brief Scale from a moving average of the range (maximum or percentile)
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from ..formats import NumberFormat
from .compute  import compute_scale
from .ema      import ema_update
from .group    import group_merge
from .minmax   import minmax_measure
from .types    import Granularity



class RangeScaler(nn.Module):
    '''
    #* Follow the range with a moving average while training, and keep it for inference.
    #? percentile=None follows the maximum ("max"). 99.9 follows the 99.9th percentile ("percentile").
    #! frozen = True stops the updates at once (capture CUDA graphs again after you change it). freeze_step stops them by itself after that many updates, with no new capture.
    '''
    
    learnable: bool = False
    kept     : torch.Tensor
    updates  : torch.Tensor
    
    def __init__(self, percentile: float | None = None, momentum: float = 0.95, scale_count: int = 1, symmetric: bool = True, freeze_step: int | None = None) -> None:
        '''
        #* Keep the settings, and start the kept range as NaN (not measured yet).
        #? scale_count is the number of scales: 1 for the tensor, or the number of channels. freeze_step is the number of updates (one for each training step) before the range stops.
        #! The count lives on the device, so stopping makes no GPU sync. A range never measured always takes the first update, so freeze_step=0 keeps the first batch.
        '''
        
        super().__init__()
        self.percentile  = percentile
        self.momentum    = momentum
        self.symmetric   = symmetric
        self.frozen      = False
        self.freeze_step = freeze_step
        self.register_buffer("kept"   , torch.full((2, scale_count), torch.nan))
        self.register_buffer("updates", torch.zeros((), dtype=torch.int64)     )
        
        return None
    
    def forward(self, x: torch.Tensor, number_format: NumberFormat, granularity: Granularity, update: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        #* Return the scale and the zero point from the kept range, after the range is updated while training.
        #! The range is NaN before the first training batch, so train or run calibrate_model() first.
        '''
        
        # Update the moving average of the range while training, and keep the old range after freeze_step updates
        if update and self.training and not self.frozen:
            current = torch.stack(minmax_measure(x, granularity, self.percentile, self.symmetric))
            moving  = ema_update(self.kept, current, self.momentum)
            if self.freeze_step is not None:
                moving = torch.where((self.updates < self.freeze_step) | torch.isnan(self.kept), moving, self.kept)
            self.kept.copy_(moving)
            self.updates.add_(1)
        
        # Turn the kept range into a scale
        scale, zero_point = compute_scale(self.kept[0], self.kept[1], number_format, self.symmetric)
        return group_merge(scale, x.shape, granularity), group_merge(zero_point, x.shape, granularity)
