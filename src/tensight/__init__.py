from .adapter   import adapter_register
from .converter import converter_register
from .core      import TensightWarning
from .inspect   import inspect_tensor
from .note      import note_tensor
from .run       import run_close, run_record_next, run_should_record, run_start, run_step
from .types     import Adapter, CodeUsage, Histogram, MapReport, PointKind, QuantInfo, QuantizerSite, QuantPair, QuantReport, QuantValue, TensorReport, ValueSummary
from .watch     import watch_model

'''
tensight/
├── __init__.py
├── types.py
├── run.py
│   ├── run_start()
│   ├── run_step()
│   ├── run_should_record()
│   ├── run_record_next()
│   ├── run_close()
│   └── run_current()
├── run_state.py
│   └── RunState
├── watch.py
│   └── watch_model()
├── note.py
│   └── note_tensor()
├── inspect.py
│   └── inspect_tensor()
├── adapter.py
│   └── adapter_register()
├── converter.py
│   └── converter_register()
├── core/        警告
├── observe/     C-02 観測
├── stats/       C-03 集計
├── adapters/    C-04 読み取り部品
├── writer/      C-05 書き込み
└── viewer/      C-06 閲覧画面
'''
