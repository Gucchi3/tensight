# @file shared_scaler.py
# @brief One scale shared by several tensors in a forward pass
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from ..formats import NumberFormat
from .compute  import compute_scale
from .ema      import ema_update
from .group    import group_merge
from .minmax   import minmax_measure
from .types    import Granularity



class SharedScaler(nn.Module):
    '''
    #* Share one scale for the whole tensor among several tensors in one forward pass (the residual stream).
    #? Call begin_forward() at the start of the forward pass of the model, and end_forward() at the end.
    #? Scale sharing of Kaneko and Meng: one SharedScaler for the residual stream of one block (its input, the branch outputs, and the adds). The block calls both around its own forward pass.
    #! The scale does not change in between, and the moving average is updated once in end_forward(). freeze_step stops the updates by itself, like RangeScaler.
    '''
    
    learnable: bool = False
    kept     : torch.Tensor
    seen     : torch.Tensor
    updates  : torch.Tensor
    
    def __init__(self, percentile: float | None = None, momentum: float = 0.95, symmetric: bool = True, freeze_step: int | None = None) -> None:
        '''
        #* Keep the settings, and start the kept range as NaN (not measured yet).
        #? freeze_step is the number of updates (forward passes) before the range stops (None never stops). A range never measured always takes the first update.
        '''
        
        super().__init__()
        self.percentile  = percentile
        self.momentum    = momentum
        self.symmetric   = symmetric
        self.frozen      = False
        self.freeze_step = freeze_step
        self.register_buffer("kept"   , torch.full((2, 1), torch.nan)     )
        self.register_buffer("seen"   , torch.full((2, 1), torch.nan)     , persistent=False)
        self.register_buffer("updates", torch.zeros((), dtype=torch.int64))
        
        return None
    
    def begin_forward(self, x: torch.Tensor | None = None) -> None:
        '''
        #* Start a forward pass: forget the ranges seen so far, and set the kept range from x if it was never set.
        #? Pass the first tensor of the stream (the output of the stem), so the very first pass also uses one scale.
        '''
        
        self.seen.fill_(torch.nan)
        if x is not None:
            current = torch.stack(minmax_measure(x, Granularity(), self.percentile, self.symmetric))
            self.kept.copy_(torch.where(torch.isnan(self.kept), current, self.kept))
        
        return None
    
    def forward(self, x: torch.Tensor, number_format: NumberFormat, granularity: Granularity, update: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        #* Return the shared scale and zero point, and remember the range of x for end_forward().
        #? While torch.export runs, it uses the kept range only (an exported model is calibrated), so the scale does not depend on x and folds into a constant.
        '''
        
        # While exporting: the kept range
        if torch.compiler.is_exporting():
            scale, zero_point = compute_scale(self.kept[0], self.kept[1], number_format, self.symmetric)
            return group_merge(scale, x.shape, granularity), group_merge(zero_point, x.shape, granularity)
        
        # Remember the widest range seen in this forward pass
        current = torch.stack(minmax_measure(x, Granularity(), self.percentile, self.symmetric))
        if update and self.training and not self.frozen:
            self.seen.copy_(torch.stack([torch.fmin(self.seen[0], current[0]), torch.fmax(self.seen[1], current[1])]))
        
        # Use the kept range, or the range of x while nothing is kept yet
        kept              = torch.where(torch.isnan(self.kept), current, self.kept)
        scale, zero_point = compute_scale(kept[0], kept[1], number_format, self.symmetric)
        return group_merge(scale, x.shape, granularity), group_merge(zero_point, x.shape, granularity)
    
    def end_forward(self) -> None:
        '''
        #* Finish a forward pass: move the kept range toward the widest range seen, and keep the old range after freeze_step updates.
        '''
        
        if self.training and not self.frozen:
            seen   = torch.where(torch.isnan(self.seen), self.kept, self.seen)
            moving = ema_update(self.kept, seen, self.momentum)
            if self.freeze_step is not None:
                moving = torch.where((self.updates < self.freeze_step) | torch.isnan(self.kept), moving, self.kept)
            self.kept.copy_(moving)
            self.updates.add_(1)
        
        return None
