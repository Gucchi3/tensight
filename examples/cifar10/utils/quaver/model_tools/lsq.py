# @file lsq.py
# @brief LSQ helpers for the optimizer
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from ..scales import STEP_MIN, LsqScaler
from .collect import collect_scalers



def lsq_parameters(model: nn.Module) -> list[nn.Parameter]:
    '''
    #* Collect the learned steps and zero points of every LsqScaler.
    '''
    
    return [parameter for _, scaler in collect_scalers(model, LsqScaler) for parameter in scaler.parameters()]


def lsq_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    '''
    #* Make the groups for the optimizer: every other parameter with weight_decay, and the LSQ steps and zero points with no weight decay.
    '''
    
    learned = lsq_parameters(model)
    chosen  = {id(parameter) for parameter in learned}
    others  = [parameter for parameter in model.parameters() if id(parameter) not in chosen]
    return [{"params": others, "weight_decay": weight_decay}, {"params": learned, "weight_decay": 0.0}]


def lsq_clamp_steps(model: nn.Module) -> None:
    '''
    #* Pull every LSQ step back above its lower limit. Call it after optimizer.step().
    #? Adam moves a step by about the learning rate, so one update can make a small step negative.
    '''
    
    with torch.no_grad():
        for _, scaler in collect_scalers(model, LsqScaler):
            scaler.step.clamp_(min=STEP_MIN)
    
    return None
