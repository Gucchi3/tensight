# @file layout.py
# @brief How a map pattern is chosen when it is "auto" (02a 4章, ADR-012)
# @author Kohei Yamaguchi, Claude

from torch import nn

from ..stats import pattern_default, pattern_parse
from ..types import MapPattern, RecordKind



LAYOUT_CONV  = (("ConvTranspose1d", "conv1d"), ("Conv1d", "conv1d"), ("ConvTranspose2d", "conv2d"), ("Conv2d", "conv2d"), ("ConvTranspose3d", "conv3d"), ("Conv3d", "conv3d"))
LAYOUT_RANK  = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d, nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d, nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
                nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d, nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d)
LAYOUT_LAST  = ("Linear", "Embedding", "LayerNorm", "RMSNorm")
RANK_PRESETS = {3: "conv1d", 4: "conv2d", 5: "conv3d"}



def layout_choose(module: nn.Module | None, kind: RecordKind, ndim: int) -> tuple[MapPattern | None, bool]:
    '''
    #* Choose the map pattern of a point whose pattern is "auto", and say whether to warn (MSG-013).
    #? Called from ModelObserver and note_tensor().
    #! The layer type is read from the end of the class names, also of the parent classes (QuantConv2d counts as Conv2d).
    '''
    
    # Scalars have no map; parameters use "weight" (02a 4章 table of auto)
    if ndim == 0:
        return None, False
    if kind in ("value", "grad", "update"):
        weight = pattern_parse("weight")
        return (weight if isinstance(weight, MapPattern) and ndim >= 2 else pattern_default(ndim)), False
    if module is None:
        return pattern_default(ndim), True
    
    # Convolutions by the end of the class name, and norms and poolings by the number of dimensions
    names  = [kind_.__name__ for kind_ in type(module).__mro__]
    preset = next((name for suffix, name in LAYOUT_CONV for class_name in names if class_name.endswith(suffix)), None)
    if preset is None and isinstance(module, LAYOUT_RANK):
        preset = RANK_PRESETS.get(ndim)
        if preset is None:
            return pattern_default(ndim), False
    
    # A preset that does not fit the number of dimensions falls back to the fixed pattern, without a warning
    if preset is not None:
        pattern = pattern_parse(preset)
        if isinstance(pattern, MapPattern) and len([name for name in pattern.left if name != "..."]) == ndim:
            return pattern, False
        return pattern_default(ndim), False
    
    # Layers that keep the features last use the fixed pattern quietly; the others warn
    known = any(class_name.endswith(suffix) for class_name in names for suffix in LAYOUT_LAST)
    return pattern_default(ndim), not known
