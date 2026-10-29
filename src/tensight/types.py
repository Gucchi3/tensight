# @file types.py
# @brief Data containers shared by every part of tensight
# @author Kohei Yamaguchi, Claude

from dataclasses import dataclass
from pathlib     import Path
from torch       import nn
from typing      import Any, Literal, Protocol

import torch



PointKind   = Literal["input", "output", "output_grad", "weight", "bias", "grad", "update"]
RecordKind  = Literal["input", "output", "output_grad", "value", "grad", "update", "note"]
RoundMode   = Literal["nearest", "floor"]
Granularity = Literal["tensor", "channel", "group"]



@dataclass(frozen=True, eq=False)
class QuantizerSite:
    '''
    #* Hold where a quantizer is, and which of its calls carries the quantization (02a 5章).
    #! method other than "forward" means that tensight wraps that method of the module.
    '''
    
    module: nn.Module
    role  : Literal["input", "weight", "output"]
    owner : str
    param : str | None = None
    method: str        = "forward"


@dataclass(frozen=True, eq=False)
class QuantInfo:
    '''
    #* Hold the quantization info of one tensor (02a 5章).
    #! levels are in code units with the zero point (INT: codes, FP: values); a real value is (level - zero_point) x scale.
    '''
    
    format     : str
    kind       : Literal["int", "fp"]
    bits       : int
    scale      : torch.Tensor
    zero_point : torch.Tensor | None
    granularity: Granularity
    codes      : torch.Tensor | None
    code_total : int
    levels     : torch.Tensor | None
    code_min   : int       = 0
    round_mode : RoundMode = "nearest"


@dataclass(frozen=True, eq=False)
class QuantValue:
    '''
    #* Hold real values and their quantization info.
    #! source is the tensor that the reader got (for example the codes of the integer mode). When it is given, the record shows its data type and saves it as the whole tensor (CR-010).
    '''
    
    values: torch.Tensor
    info  : QuantInfo
    source: torch.Tensor | None = None


@dataclass(frozen=True, eq=False)
class QuantPair:
    '''
    #* Hold the values before and after a quantizer, with the quantization info.
    #! after is None when the quantized values come later as the values of the point (03 U-027).
    '''
    
    before: torch.Tensor
    after : torch.Tensor | None
    info  : QuantInfo


class Adapter(Protocol):
    '''
    #* The shape of a reader for a quantization library (02a 5章).
    #? adapter_register() checks that an object has these names.
    '''
    
    name: str
    
    def find_quantizers(self, model: nn.Module) -> list[QuantizerSite]:
        '''
        #* Find the quantizers in the model.
        '''
        
        ...
    
    def read_value(self, value: object) -> QuantValue | None:
        '''
        #* Read a value of a layer, or return None if this reader does not handle it.
        '''
        
        ...
    
    def read_quantizer(self, site: QuantizerSite, args: tuple[Any, ...], result: object) -> QuantPair | None:
        '''
        #* Read one call of a quantizer, or return None if this reader does not handle it.
        '''
        
        ...


@dataclass(frozen=True)
class Histogram:
    '''
    #* Hold the distribution of the values (FR-009).
    '''
    
    lo    : float | None
    hi    : float | None
    counts: tuple[int, ...]


@dataclass(frozen=True)
class ValueSummary:
    '''
    #* Hold the count, the smallest, the largest, and the mean of the scales or the zero points.
    '''
    
    count: int
    min  : float
    max  : float
    mean : float


@dataclass(frozen=True)
class CodeUsage:
    '''
    #* Hold how the codes are used (FR-014).
    '''
    
    total : int
    used  : int
    offset: int
    counts: tuple[int, ...] | None


@dataclass(frozen=True)
class QuantReport:
    '''
    #* Hold the summary of the quantization of one tensor (FR-012 to FR-015).
    '''
    
    format     : str
    kind       : str
    bits       : int
    scale      : ValueSummary
    zero_point : ValueSummary | None
    granularity: str
    codes      : CodeUsage | None
    levels     : tuple[float, ...] | None
    round_mode : str
    clip       : float | None
    mse        : float | None
    max_abs    : float | None
    sqnr       : float | None


@dataclass(frozen=True, eq=False)
class MapReport:
    '''
    #* Hold the map of absolute values of one tensor (FR-035).
    '''
    
    layout  : str
    pattern : str
    rows    : int
    cols    : int
    row_size: int
    col_size: int
    row     : tuple[str, ...]
    col     : tuple[str, ...]
    max     : float
    values  : torch.Tensor


@dataclass(frozen=True, eq=False)
class TensorReport:
    '''
    #* Hold the summary of one tensor: what inspect_tensor() returns, and what one record holds (02a API-09).
    '''
    
    dtype : str
    shape : tuple[int, ...]
    device: str
    numel : int
    min   : float | None
    max   : float | None
    mean  : float | None
    std   : float | None
    l2    : float | None
    zero  : float
    nan   : int
    inf   : int
    hist  : Histogram
    sample: tuple[float, ...]
    quant : QuantReport | None
    map   : MapReport | None


@dataclass(frozen=True)
class MapPattern:
    '''
    #* Hold a parsed pattern of a map (02a 4章).
    #! "..." stands for any number of dimensions.
    '''
    
    text  : str
    preset: str | None
    left  : tuple[str, ...]
    groups: tuple[tuple[str, ...], ...]


MapSpec = MapPattern | Literal["auto"]


@dataclass(frozen=True, eq=False)
class RunSettings:
    '''
    #* Hold the settings of one run (02a API-01).
    #? run_start() makes this after it checks the arguments.
    '''
    
    log_dir       : Path
    name          : str | None
    points        : tuple[str, ...] | None
    kinds         : tuple[PointKind, ...]
    interval      : int
    bins          : int
    sample_size   : int
    boundary_width: float
    track_elements: dict[str, tuple[int, ...]] | None
    compare_steps : frozenset[int]
    save_full     : dict[str, frozenset[int]]
    maps          : tuple[tuple[str, MapSpec], ...]


@dataclass(frozen=True)
class PointInfo:
    '''
    #* Hold one observation point (a point line of catalog.js).
    '''
    
    name  : str
    target: str
    kind  : RecordKind
    layer : str | None


@dataclass(frozen=True, eq=False)
class StepRecord:
    '''
    #* Hold one record of one point: the collector makes it, and the writer writes it.
    #! boundary keeps NaN where there is no value.
    '''
    
    point   : PointInfo
    report  : TensorReport
    elements: dict[int, tuple[float, float | None, float | None]] | None
    boundary: tuple[float, float, float] | None
    ratio   : float | None
