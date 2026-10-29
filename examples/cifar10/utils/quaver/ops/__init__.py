from .accumulate import ACCUMULATOR_FORMAT, WIDE_ACCUMULATOR_FORMAT, accumulate_bits, accumulate_conv2d, accumulate_conv2d_device, accumulate_format, accumulate_linear, accumulate_linear_device, accumulate_matmul, accumulate_matmul_device
from .accumulate import accumulate_product, accumulate_scale
from .activation import activation_device, activation_fake, activation_float, activation_integer
from .add        import add_device, add_fake, add_float, add_integer
from .bias       import bias_quantize
from .block      import block_quantize_device, block_quantize_fake, block_quantize_float, block_quantize_integer
from .matmul     import matmul_fake, matmul_float, matmul_integer
from .mode       import FAKE_OPS, FLOAT_OPS, INTEGER_OPS, mode_ops
from .normalize  import NORM_BITS, normalize_l1_device, normalize_l1_fake, normalize_l1_float, normalize_l1_integer, normalize_l1_scores_device, normalize_l1_scores_fake, normalize_l1_scores_float, normalize_l1_scores_integer
from .pool       import pool_avg2d_device, pool_avg2d_fake, pool_avg2d_float, pool_avg2d_integer
from .requantize import REQUANTIZE_REGISTRY, build_requantize, requantize_device, requantize_fake, requantize_float, requantize_integer, requantize_params, requantize_ratio, requantize_rescale
from .types      import ActivationKind, FixedPoint, NeqScope, OpSet, RequantizeStyle
from .wide       import wide_round, wide_round_format

'''
ops/
├── __init__.py
├── accumulate.py
│   ├── accumulate_scale()
│   ├── accumulate_format()
│   ├── accumulate_bits()
│   ├── accumulate_conv2d()
│   ├── accumulate_linear()
│   ├── accumulate_matmul()
│   ├── accumulate_conv2d_device()
│   ├── accumulate_linear_device()
│   ├── accumulate_matmul_device()
│   └── accumulate_product()
├── activation.py
│   ├── activation_fake()
│   ├── activation_integer()
│   ├── activation_device()
│   └── activation_float()
├── add.py
│   ├── add_fake()
│   ├── add_integer()
│   ├── add_device()
│   └── add_float()
├── bias.py
│   └── bias_quantize()
├── block.py
│   ├── block_quantize_fake()
│   ├── block_quantize_integer()
│   ├── block_quantize_device()
│   └── block_quantize_float()
├── matmul.py
│   ├── matmul_fake()
│   ├── matmul_integer()
│   └── matmul_float()
├── mode.py
│   └── mode_ops()
├── normalize.py
│   ├── normalize_l1_fake()
│   ├── normalize_l1_integer()
│   ├── normalize_l1_device()
│   ├── normalize_l1_float()
│   ├── normalize_l1_scores_fake()
│   ├── normalize_l1_scores_integer()
│   ├── normalize_l1_scores_device()
│   └── normalize_l1_scores_float()
├── pool.py
│   ├── pool_avg2d_fake()
│   ├── pool_avg2d_integer()
│   ├── pool_avg2d_device()
│   └── pool_avg2d_float()
├── requantize.py
│   ├── build_requantize()
│   ├── requantize_params()
│   ├── requantize_ratio()
│   ├── requantize_rescale()
│   ├── requantize_fake()
│   ├── requantize_integer()
│   ├── requantize_device()
│   └── requantize_float()
├── types.py
│   ├── ActivationKind
│   ├── NeqScope
│   ├── RequantizeStyle
│   ├── FixedPoint
│   └── OpSet
└── wide.py
    ├── wide_round()
    └── wide_round_format()
'''
