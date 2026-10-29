from .cifar10    import cifar10_augment, cifar10_load
from .exceptions import Error
from .train      import BATCH_SIZE, train_evaluate, train_fit, train_step

'''
utils/
├── __init__.py
├── cifar10.py
│   ├── cifar10_load()
│   └── cifar10_augment()
├── exceptions.py
│   └── Error
├── train.py
│   ├── train_fit()
│   ├── train_step()
│   └── train_evaluate()
└── quaver/          (a copy of quaver: https://github.com/Gucchi3/quaver)
'''
