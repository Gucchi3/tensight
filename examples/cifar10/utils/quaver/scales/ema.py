# @file ema.py
# @brief Moving average that starts from NaN
# @author Kohei Yamaguchi, Claude

import torch



def ema_update(running: torch.Tensor, current: torch.Tensor, momentum: float) -> torch.Tensor:
    '''
    #* Return the moving average of running and current.
    #! Where running is NaN (never measured), return current as it is. This uses torch.where, so there is no GPU sync.
    '''
    
    average = running * momentum + current * (1.0 - momentum)
    return torch.where(torch.isnan(running), current, average)
