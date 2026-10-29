# @file freeze.py
# @brief Stop or start again the range updates of the scalers
# @author Kohei Yamaguchi, Claude

from torch import nn

from ..scales import RangeScaler, SharedScaler
from .collect import collect_scalers



def freeze_scalers(model: nn.Module, frozen: bool = True, step: int | None = None) -> int:
    '''
    #* Stop (frozen=True) or start again (frozen=False) the range updates of the max, percentile, and shared scalers, or with step, make them stop by themselves later. Returns how many scalers it set.
    #? Without step, call it after a few epochs (range_freeze_epoch in E_AI_deit). With step, call it once before training, with step = epochs x steps per epoch.
    #! Capture CUDA graphs again after you change frozen. A step set before the capture needs no new capture, because the stop happens on the device.
    '''
    
    scalers = [*collect_scalers(model, RangeScaler), *collect_scalers(model, SharedScaler)]
    for _, scaler in scalers:
        if step is None:
            scaler.frozen = frozen
        else:
            scaler.freeze_step = step
    return len(scalers)
