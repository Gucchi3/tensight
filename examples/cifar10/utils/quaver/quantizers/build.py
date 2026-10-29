# @file build.py
# @brief Build quantizers from QuantizerSpec
# @author Kohei Yamaguchi, Claude

##########################################################
'''
Add a scaler
1. Write build_xxx_scaler(spec, scale_count) to return a scaler.
2. Add "name: build_xxx_scaler" to SCALER_REGISTRY.
3. Set QuantizerSpec.scaler to that name.
'''
##########################################################

from collections.abc import Callable

from ..formats  import build_format
from ..scales   import BlockScaler, DynamicScaler, Granularity, LsqScaler, RangeScaler, SharedScaler
from .quantizer import Quantizer
from .types     import QuantizerSpec, Scaler



def build_dynamic_scaler(spec: QuantizerSpec, scale_count: int) -> Scaler:
    '''
    #* Build a scaler that measures the current tensor every time.
    '''
    
    return DynamicScaler(percentile=None, symmetric=spec.symmetric)


def build_max_scaler(spec: QuantizerSpec, scale_count: int) -> Scaler:
    '''
    #* Build a scaler that follows the maximum with a moving average.
    '''
    
    return RangeScaler(percentile=None, momentum=spec.momentum, scale_count=scale_count, symmetric=spec.symmetric, freeze_step=spec.freeze_step)


def build_percentile_scaler(spec: QuantizerSpec, scale_count: int) -> Scaler:
    '''
    #* Build a scaler that follows a percentile (99.9 by default) with a moving average.
    '''
    
    return RangeScaler(percentile=spec.percentile, momentum=spec.momentum, scale_count=scale_count, symmetric=spec.symmetric, freeze_step=spec.freeze_step)


def build_lsq_scaler(spec: QuantizerSpec, scale_count: int) -> Scaler:
    '''
    #* Build a scaler that learns the step (LSQ, or LSQ+ when asymmetric).
    '''
    
    return LsqScaler(scale_count=scale_count, symmetric=spec.symmetric, init=spec.lsq_init)


def build_shared_scaler(spec: QuantizerSpec, scale_count: int) -> Scaler:
    '''
    #* Build a scaler that several quantizers can share.
    #! Each call makes a new one. To share it, make one SharedScaler and pass it to each Quantizer.
    '''
    
    return SharedScaler(percentile=None, momentum=spec.momentum, symmetric=spec.symmetric, freeze_step=spec.freeze_step)


def build_nvfp4_scaler(spec: QuantizerSpec, scale_count: int) -> Scaler:
    '''
    #* Build the scaler with the two levels of NVFP4: a UE4M3 scale for each block, and a scale for the tensor measured every time.
    '''
    
    return BlockScaler("ue4m3")


SCALER_REGISTRY: dict[str, Callable[[QuantizerSpec, int], Scaler]] = {
    "dynamic"   : build_dynamic_scaler,
    "max"       : build_max_scaler,
    "percentile": build_percentile_scaler,
    "lsq"       : build_lsq_scaler,
    "shared"    : build_shared_scaler,
    "nvfp4"     : build_nvfp4_scaler,
}


def build_quantizer(spec: QuantizerSpec, scale_count: int = 1) -> Quantizer:
    '''
    #* Build a Quantizer from a QuantizerSpec.
    #? scale_count is the number of scales for max, percentile, and lsq (for example, the number of channels).
    #! Add new scalers to SCALER_REGISTRY.
    '''
    
    # Find the builder of the scaler
    if spec.scaler not in SCALER_REGISTRY:
        raise KeyError(f"Unknown scaler {spec.scaler!r}. Available: {', '.join(SCALER_REGISTRY)}")
    
    # Put the parts together
    scaler      = SCALER_REGISTRY[spec.scaler](spec, scale_count)
    granularity = Granularity(spec.granularity, spec.axis, spec.block_size)
    return Quantizer(build_format(spec.format), scaler, granularity, spec.round_mode, spec.pow2, spec.batch_axis)
