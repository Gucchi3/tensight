# @file watch.py
# @brief Put the observation points on a model (02a API-02)
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from .adapters import registry_adapters
from .core     import message_warn
from .observe  import ModelObserver
from .run      import run_current



def watch_model(model: nn.Module, optimizer: torch.optim.Optimizer | None = None) -> list[str]:
    '''
    #* Put the hooks of the observation points on a model and its optimizer (02a API-02, IF-01).
    #? A module made by torch.compile keeps the model in _orig_mod; the hooks go on that model, so the names of the points stay the same (ADR-016).
    #! Call it after run_start(). Watching the same model again takes the old hooks off first.
    '''
    
    # Check the run and the arguments
    state = run_current()
    if state is None:
        raise RuntimeError("watch_model の前に run_start を呼んでください")
    if not isinstance(model, nn.Module):
        raise TypeError("引数 model は nn.Module にしてください。渡された値: " + repr(model))
    if optimizer is not None and not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("引数 optimizer は torch.optim.Optimizer か None にしてください。渡された値: " + repr(optimizer))
    
    # Watch the model inside a module made by torch.compile
    original = getattr(model, "_orig_mod", None)
    if isinstance(original, nn.Module):
        model = original
    
    # Put the hooks, warn when there is no point to record (CR-019), and return the names of the points
    observer = ModelObserver(model, optimizer, state.settings, state.collector, registry_adapters())
    state.watch(observer)
    if not observer.names:
        message_warn("MSG-015")
    return list(observer.names)
