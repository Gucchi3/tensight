from .quaver_adapter import QuaverAdapter
from .registry       import ADAPTER_REGISTRY, registry_adapters, registry_add
from .torch_adapter  import TorchAdapter

'''
adapters/
├── __init__.py
├── registry.py
│   ├── registry_add()
│   └── registry_adapters()
├── torch_adapter.py
│   └── TorchAdapter
└── quaver_adapter.py
    └── QuaverAdapter
'''
