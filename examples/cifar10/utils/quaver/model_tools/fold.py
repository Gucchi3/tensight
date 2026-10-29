# @file fold.py
# @brief Fold BatchNorm into the conv before it
# @author Kohei Yamaguchi, Claude

from torch                 import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

import copy

FOLD_PAIRS = ((nn.Conv1d, nn.BatchNorm1d), (nn.Conv2d, nn.BatchNorm2d), (nn.Conv3d, nn.BatchNorm3d))



def fold_batch_norms(model: nn.Module) -> nn.Module:
    '''
    #* Return a copy of the model in eval mode, with each BatchNorm folded into the conv just before it.
    #! It only folds a conv and a BatchNorm that are next to each other in the same parent module.
    '''
    
    # Look at the children of every module in the copy
    folded = copy.deepcopy(model).eval()
    for module in list(folded.modules()):
        children = list(module.named_children())
        
        # Swap each (conv, BatchNorm) pair for the folded conv and an Identity
        for (conv_name, conv), (norm_name, norm) in zip(children, children[1:]):
            for conv_type, norm_type in FOLD_PAIRS:
                if isinstance(conv, conv_type) and isinstance(norm, norm_type):
                    setattr(module, conv_name, fuse_conv_bn_eval(conv, norm))
                    setattr(module, norm_name, nn.Identity()                )
    return folded
