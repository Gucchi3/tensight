from .calibrate import calibrate_model
from .collect   import collect_quantizers, collect_scalers
from .fold      import fold_batch_norms
from .freeze    import freeze_scalers
from .lsq       import lsq_clamp_steps, lsq_param_groups, lsq_parameters
from .precision import MATH_TYPES, precision_disable_tf32, precision_set_math
from .switch    import switch_integer_mode, switch_quantizers
from .sync      import sync_count

'''
model_tools/
├── __init__.py
├── calibrate.py
│   └── calibrate_model()
├── collect.py
│   ├── collect_quantizers()
│   └── collect_scalers()
├── fold.py
│   └── fold_batch_norms()
├── freeze.py
│   └── freeze_scalers()
├── lsq.py
│   ├── lsq_parameters()
│   ├── lsq_param_groups()
│   └── lsq_clamp_steps()
├── precision.py
│   ├── precision_disable_tf32()
│   └── precision_set_math()
├── switch.py
│   ├── switch_quantizers()
│   └── switch_integer_mode()
└── sync.py
    └── sync_count()
'''
