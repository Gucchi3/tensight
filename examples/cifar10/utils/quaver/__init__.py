from .export      import model_export
from .formats     import build_format
from .model_tools import calibrate_model, fold_batch_norms, freeze_scalers, lsq_clamp_steps, lsq_param_groups, precision_disable_tf32, precision_set_math, switch_integer_mode, switch_quantizers
from .quantizers  import Quantizer, QuantizerSpec, build_quantizer
from .samples     import QuantAdd, QuantAvgPool2d, QuantBlockConv2d, QuantBlockLinear, QuantConv2d, QuantLinear, QuantSimaAttention
from .scales      import BlockScaler, DynamicScaler, Granularity, LsqScaler, QuantParams, RangeScaler, SharedScaler, params_codes, params_values

'''
quaver/
├── __init__.py    # the basic API: sample layers, quantizers, and whole-model tools (every other public name is in its folder)
├── formats/       # number formats (INT, FP, UFP), exact rounding, and the registration of the device ops (torch.ops.quaver)
├── scales/        # how scales are chosen: dynamic, max, percentile, LSQ, shared, block scales (NVFP4, integer)
├── quantizers/    # Quantizer (fake quantization), LSQ, build from QuantizerSpec
├── ops/           # quantized ops in the training mode and the integer mode (the integer steps are device ops)
├── model_tools/   # whole-model helpers: collect, switch, freeze, calibrate, LSQ, fold, TF32, syncs
├── samples/       # sample layers: Conv2d, Linear, block-scaled Conv2d and Linear (NVFP4 and the like), SimA attention, residual add, average pooling
└── export/        # export a model in the integer mode for the device: model.json and model.bin (needs numpy)
'''
