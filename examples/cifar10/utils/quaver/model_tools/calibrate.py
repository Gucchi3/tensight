# @file calibrate.py
# @brief Measure the ranges of the activations before training, or for PTQ
# @author Kohei Yamaguchi, Claude

from collections.abc import Callable, Iterable
from torch           import nn

import torch

from .collect import collect_scalers
from .switch  import switch_quantizers



def calibrate_model(model: nn.Module, batches: Iterable, forward: Callable | None = None) -> None:
    '''
    #* Run batches through the model in float (quantization off), so that the scalers learn the ranges of the float values, like the observers of PyTorch.
    #? Use it for PTQ, or before QAT to start from good scales. LSQ steps start from the first batch. Layers with a float_mode flag compute in float, so no value is rounded or clipped.
    #! Pass forward(model, batch) when a batch is not one input tensor.
    '''
    
    # Measure in eval mode, but let the scalers update (switch_quantizers() also puts the layers into the float mode)
    was_training = model.training
    model.eval()
    for _, scaler in collect_scalers(model, nn.Module):
        scaler.train()
    switch_quantizers(model, False)
    
    # Run the batches without gradients
    with torch.no_grad():
        for batch in batches:
            if forward is None:
                model(batch)
            else:
                forward(model, batch)
    
    # Put everything back
    switch_quantizers(model, True)
    model.train(was_training)
    
    return None
