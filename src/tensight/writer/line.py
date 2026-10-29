# @file line.py
# @brief One line of the record folder, and the items of one record (02c)
# @author Kohei Yamaguchi, Claude

from base64          import b64encode
from collections.abc import Mapping

import json
import math
import torch

from ..types import MapPattern, RunSettings, StepRecord



FORMAT = 1



def line_encode(function: str, payload: Mapping[str, object]) -> str:
    '''
    #* Make one line "tensight.function({...});" of the record folder (02c, ADR-002).
    #! A float that is not finite raises ValueError, so pass the items of line_record().
    '''
    
    return "tensight." + function + "(" + json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(", ", ": ")) + ");\n"


def line_record(run: str, point_id: int, step: int, time: str, record: StepRecord, levels: bool) -> dict[str, object]:
    '''
    #* Turn one record into the items of a rec line, in the order of 02c.
    #? Called from RecordWriter. levels says whether this record carries the list of levels.
    '''
    
    # Put the basic items, the statistics, the distribution and the first values (FR-008 to FR-010)
    report = record.report
    
    # Put them in the order of the table of 02c
    line: dict[str, object] = {
        "format": FORMAT,
        "run"   : run,
        "id"    : point_id,
        "step"  : step,
        "time"  : time,
        "dtype" : report.dtype,
        "shape" : list(report.shape),
        "device": report.device,
        "numel" : report.numel,
        "min"   : _line_number(report.min),
        "max"   : _line_number(report.max),
        "mean"  : _line_number(report.mean),
        "std"   : _line_number(report.std),
        "l2"    : _line_number(report.l2),
        "zero"  : report.zero,
        "nan"   : report.nan,
        "inf"   : report.inf,
        "hist"  : {"lo": _line_number(report.hist.lo), "hi": _line_number(report.hist.hi), "counts": list(report.hist.counts)},
        "sample": [_line_text(v) for v in report.sample],
    }
    
    # Put the followed elements with their scales and zero points (FR-032)
    if record.elements is not None:
        line["elements"] = {str(i): {"value": _line_text(v), "scale": _line_number(s), "zero_point": _line_number(z)} for i, (v, s, z) in record.elements.items()}
    
    # Put the quantization info (FR-012 to FR-015)
    quant = report.quant
    if quant is not None:
        line["quant"] = {
            "format"     : quant.format,
            "kind"       : quant.kind,
            "bits"       : quant.bits,
            "scale"      : {"count": quant.scale.count, "min": quant.scale.min, "max": quant.scale.max, "mean": quant.scale.mean},
            "zero_point" : None if quant.zero_point is None else {"count": quant.zero_point.count, "min": quant.zero_point.min, "max": quant.zero_point.max, "mean": quant.zero_point.mean},
            "granularity": quant.granularity,
            "codes"      : None if quant.codes is None else {"total": quant.codes.total, "used": quant.codes.used, "offset": quant.codes.offset, "counts": None if quant.codes.counts is None else list(quant.codes.counts)},
            "levels"     : list(quant.levels) if levels and quant.levels is not None else None,
            "round_mode" : quant.round_mode,
            "clip"       : _line_number(quant.clip),
            "mse"        : _line_number(quant.mse),
            "max_abs"    : _line_number(quant.max_abs),
            "sqnr"       : "inf" if quant.sqnr == math.inf else _line_number(quant.sqnr),
        }
    
    # Put the boundary ratios (FR-031) and the update ratio (FR-028)
    if record.boundary is not None:
        near, changed, back = record.boundary
        line["boundary"]    = {"near": _line_number(near), "changed": _line_number(changed), "back": _line_number(back)}
    if record.point.kind == "update":
        line["ratio"] = _line_number(record.ratio)
    
    # Put the map as bytes, where 255 means the largest cell (FR-035, ADR-009)
    map_ = report.map
    if map_ is not None:
        cells       = torch.round(map_.values.double() * 255.0 / map_.max) if map_.max > 0 else torch.zeros_like(map_.values, dtype=torch.float64)
        line["map"] = {
            "layout"  : map_.layout,
            "pattern" : map_.pattern,
            "rows"    : map_.rows,
            "cols"    : map_.cols,
            "row_size": map_.row_size,
            "col_size": map_.col_size,
            "row"     : list(map_.row),
            "col"     : list(map_.col),
            "max"     : map_.max,
            "data"    : b64encode(bytes(cells.to(torch.uint8).reshape(-1).tolist())).decode("ascii"),
        }
    return line


def line_settings(settings: RunSettings) -> dict[str, object]:
    '''
    #* Turn the settings into the items of "settings" in run.js (02c).
    '''
    
    # Write each map pattern as the preset name, auto, or the pattern itself
    maps = {key: spec.preset or spec.text if isinstance(spec, MapPattern) else spec for key, spec in settings.maps}
    
    # Put the settings in the order of 02c
    return {
        "interval"      : settings.interval,
        "bins"          : settings.bins,
        "sample_size"   : settings.sample_size,
        "boundary_width": settings.boundary_width,
        "kinds"         : list(settings.kinds),
        "points"        : None if settings.points is None else list(settings.points),
        "track_elements": None if settings.track_elements is None else {key: list(value) for key, value in settings.track_elements.items()},
        "maps"          : maps,
        "compare_steps" : sorted(settings.compare_steps),
        "save_full"     : {key: sorted(value) for key, value in settings.save_full.items()},
    }


def _line_number(value: float | None) -> float | None:
    '''
    #* Return a finite number as it is, and None for None, NaN, and Inf.
    #? line_record() uses this for numbers that JSON cannot hold.
    '''
    
    return value if value is not None and math.isfinite(value) else None


def _line_text(value: float) -> float | str:
    '''
    #* Return a finite number as it is, and "nan", "inf", or "-inf" for the others (02c sample).
    '''
    
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value
