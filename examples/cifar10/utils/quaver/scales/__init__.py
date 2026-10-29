from .block_scaler   import BlockScaler
from .compute        import SCALE_MIN, compute_block, compute_mse_range, compute_pow2, compute_scale
from .dynamic_scaler import DynamicScaler, dynamic_scaler_device
from .ema            import ema_update
from .group          import group_count, group_merge, group_split
from .lsq_scaler     import STEP_MIN, LsqScaler
from .minmax         import minmax_measure
from .params         import MATH_DTYPES, params_attach, params_bits, params_codes, params_integers, params_math_dtype, params_named, params_note, params_note_device, params_round, params_unit, params_values
from .range_scaler   import RangeScaler
from .shared_scaler  import SharedScaler
from .types          import BlockScale, Granularity, GranularityKind, LsqInit, MathDtype, QuantParams

'''
scales/
├── __init__.py
├── block_scaler.py
│   └── BlockScaler
│       ├── forward()
│       ├── split()
│       └── tensor_scale()
├── compute.py
│   ├── compute_scale()
│   ├── compute_mse_range()
│   ├── compute_pow2()
│   └── compute_block()
├── dynamic_scaler.py
│   ├── DynamicScaler
│   │   └── forward()
│   └── dynamic_scaler_device()
├── ema.py
│   └── ema_update()
├── group.py
│   ├── group_count()
│   ├── group_split()
│   └── group_merge()
├── lsq_scaler.py
│   └── LsqScaler
│       └── forward()
├── minmax.py
│   └── minmax_measure()
├── params.py
│   ├── params_math_dtype()
│   ├── params_codes()
│   ├── params_values()
│   ├── params_unit()
│   ├── params_integers()
│   ├── params_bits()
│   ├── params_named()
│   ├── params_note()
│   ├── params_note_device()
│   ├── params_round()
│   └── params_attach()
├── range_scaler.py
│   └── RangeScaler
│       └── forward()
├── shared_scaler.py
│   └── SharedScaler
│       ├── begin_forward()
│       ├── forward()
│       └── end_forward()
└── types.py
    ├── GranularityKind
    ├── LsqInit
    ├── MathDtype
    ├── Granularity
    ├── BlockScale
    └── QuantParams
        └── code_scale()
'''
