# @file quant.py
# @brief The summary of quantization, and the moves near the boundaries (FR-012 to FR-015, FR-031)
# @author Kohei Yamaguchi, Claude

import math
import torch

from ..types  import QuantInfo, QuantPair, QuantValue
from .summary import SUMMARY_CHUNK
from .types   import PendingQuant



CODES_MAX = 4096

_levels_cache: dict[tuple[str, int, float, float, str], torch.Tensor] = {}



def quant_summary(quant: QuantValue | QuantPair, values: torch.Tensor) -> PendingQuant:
    '''
    #* Summarize the scales, the zero points, the codes, the clipping and the errors on the device (03 U-025).
    #? Called from StepCollector and inspect_tensor().
    #! x = before / scale + zero point is computed in float64 with the zero point of each element, so per-channel zero points are exact.
    '''
    
    # Summarize the scales and the zero points: count, min, max, mean (FR-012)
    info      = quant.info
    device    = values.device
    double    = torch.float64
    scale     = info.scale.detach().to(device=device, dtype=double)
    zero      = info.zero_point.detach().to(device=device, dtype=double) if info.zero_point is not None else None
    scale_sum = torch.stack([torch.full((), float(scale.numel()), dtype=double, device=device), scale.min(), scale.max(), scale.mean()])
    zero_sum  = torch.stack([torch.full((), float(zero.numel()), dtype=double, device=device), zero.min(), zero.max(), zero.mean()]) if zero is not None else None
    counts    = None
    used      = None
    
    # Count the codes; large formats keep only the number of different codes (FR-014)
    if info.codes is not None:
        codes = info.codes.detach().reshape(-1).to(device)
        if info.code_total <= CODES_MAX:
            position = codes - info.code_min
            inside   = (position >= 0) & (position < info.code_total)
            counts   = torch.zeros(info.code_total + 1, dtype=torch.int64, device=device)
            counts.scatter_add_(0, torch.where(inside, position, info.code_total), torch.ones(1, dtype=torch.int64, device=device).expand(codes.numel()))
            counts   = counts[:info.code_total]
            used     = (counts > 0).sum()
        else:
            ordered = codes.sort().values
            used    = (ordered[1:] != ordered[:-1]).sum() + 1 if codes.numel() > 0 else torch.zeros((), dtype=torch.int64, device=device)
    
    # Without the values before the quantizer, there is no clipping and no error
    clip   = None
    errors = None
    if not isinstance(quant, QuantPair):
        return PendingQuant(info.format, info.kind, info.bits, info.granularity, info.code_total, info.code_min, info.round_mode, scale_sum, zero_sum, counts, used, info.levels, clip, errors)
    before = quant.before.detach().to(device)
    after  = values if quant.after is None else quant.after.detach().to(device)
    if before.shape != after.shape:
        return PendingQuant(info.format, info.kind, info.bits, info.granularity, info.code_total, info.code_min, info.round_mode, scale_sum, zero_sum, counts, used, info.levels, clip, errors)
    
    # Find the range of codes that the rounding keeps: the ends of the levels, with half a step (nearest) or one step (floor)
    edges = None
    if info.levels is not None and info.levels.numel() > 0:
        levels = info.levels.detach().to("cpu", double)
        first  = float(levels[0])
        last   = float(levels[-1])
        gap0   = float(levels[1] - levels[0]) if levels.numel() > 1 else 1.0
        gap1   = float(levels[-1] - levels[-2]) if levels.numel() > 1 else 1.0
        edges  = (first - gap0 / 2, last + gap1 / 2) if info.round_mode == "nearest" else (first, last + gap1)
    elif info.kind == "int":
        first = float(info.code_min)
        last  = float(info.code_min + info.code_total - 1)
        edges = (first - 0.5, last + 0.5) if info.round_mode == "nearest" else (first, last + 1.0)
    
    # Go through the first dimension in parts, with the scales and zero points broadcast to each part
    scale_all = torch.broadcast_to(scale, before.shape)
    zero_all  = torch.broadcast_to(zero, before.shape) if zero is not None else None
    rows      = before.shape[0] if before.ndim > 0 else 1
    step      = max(1, SUMMARY_CHUNK // max(1, before.numel() // max(1, rows)))
    outside   = torch.zeros((), dtype=torch.int64, device=device)
    finite    = torch.zeros((), dtype=torch.int64, device=device)
    error2    = torch.zeros((), dtype=double, device=device)
    error_max = torch.zeros((), dtype=double, device=device)
    signal2   = torch.zeros((), dtype=double, device=device)
    paired    = torch.zeros((), dtype=torch.int64, device=device)
    for start in range(0, rows, step):
        part_b = before[start:start + step] if before.ndim > 0 else before
        part_a = after[start:start + step] if after.ndim > 0 else after
        
        # Clipping: x in code units (with the zero point) outside the kept range (FR-013)
        if edges is not None:
            x        = part_b.to(double) / (scale_all[start:start + step] if before.ndim > 0 else scale_all)
            x        = x + (zero_all[start:start + step] if before.ndim > 0 else zero_all) if zero_all is not None else x
            ok       = torch.isfinite(x)
            kept     = (x >= edges[0]) & (x <= edges[1]) if info.round_mode == "nearest" else (x >= edges[0]) & (x < edges[1])
            outside += (ok & ~kept).sum()
            finite  += ok.sum()
        
        # Errors between after and before, in float64, for the elements where both are finite (FR-015)
        both      = torch.isfinite(part_a) & torch.isfinite(part_b)
        error     = torch.where(both, part_a.to(double) - part_b.to(double), 0)
        signal    = torch.where(both, part_b.to(double), 0)
        error2   += (error * error).sum()
        error_max = torch.maximum(error_max, error.abs().amax())
        signal2  += (signal * signal).sum()
        paired   += both.sum()
    clip   = torch.stack([outside, finite]).to(double) if edges is not None else None
    errors = torch.stack([error2, error_max, signal2, paired.to(double)])
    return PendingQuant(info.format, info.kind, info.bits, info.granularity, info.code_total, info.code_min, info.round_mode, scale_sum, zero_sum, counts, used, info.levels, clip, errors)


def quant_boundary(before: torch.Tensor, info: QuantInfo, codes: torch.Tensor, history: tuple[torch.Tensor | None, torch.Tensor | None], width: float) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor | None]]:
    '''
    #* Compute the ratio near a boundary, the ratio whose code changed, and the ratio that went back (FR-031, 03 U-026).
    #? Called from StepCollector for quantized parameters. history holds the codes of the last record and of the one before it.
    #! The boundaries are the midpoints of the levels (nearest) or the levels themselves (floor). NaN means "なし".
    '''
    
    # Keep NaN where there is no value
    device = before.device
    double = torch.float64
    count  = before.numel()
    near   = torch.full((), math.nan, dtype=double, device=device)
    
    # Near a boundary: the distance to the nearest boundary is at most width x the gap of that boundary
    if info.levels is not None and info.levels.numel() >= 2 and count > 0:
        
        # Put the levels on the device once for each format (a copy to the GPU waits for it)
        key    = (info.format, info.levels.numel(), float(info.levels[0]), float(info.levels[-1]), str(device))
        if key not in _levels_cache:
            _levels_cache[key] = info.levels.detach().to(device=device, dtype=double)
        levels = _levels_cache[key]
        bounds = (levels[:-1] + levels[1:]) / 2 if info.round_mode == "nearest" else levels[1:]
        gap    = levels[1:] - levels[:-1]
        last   = bounds.numel() - 1
        
        # Go through the first dimension in parts, in code units with the zero point
        scale_all = torch.broadcast_to(info.scale.detach().to(device=device, dtype=double), before.shape)
        zero_all  = torch.broadcast_to(info.zero_point.detach().to(device=device, dtype=double), before.shape) if info.zero_point is not None else None
        rows      = before.shape[0] if before.ndim > 0 else 1
        step      = max(1, SUMMARY_CHUNK // max(1, count // max(1, rows)))
        hits      = torch.zeros((), dtype=torch.int64, device=device)
        for start in range(0, rows, step):
            part     = before[start:start + step] if before.ndim > 0 else before
            x        = part.to(double) / (scale_all[start:start + step] if before.ndim > 0 else scale_all)
            x        = x + (zero_all[start:start + step] if before.ndim > 0 else zero_all) if zero_all is not None else x
            found    = torch.searchsorted(bounds, x.contiguous())
            left     = (found - 1).clamp(0, last)
            right    = found.clamp(0, last)
            to_left  = (x - bounds[left]).abs()
            to_right = (x - bounds[right]).abs()
            pick     = to_left <= to_right
            hits    += (torch.where(pick, to_left, to_right) <= width * torch.where(pick, gap[left], gap[right])).sum()
        near      = hits.to(double) / count
    
    # Name the codes of the last record and of the one before it
    last_codes, older = history
    
    # Changed since the last record, and back to the code of the record before it
    flat    = codes.detach().reshape(-1).to(device)
    nan     = torch.full((), math.nan, dtype=double, device=device)
    changed = (flat != last_codes).sum().to(double) / count if last_codes is not None and last_codes.shape == flat.shape and count > 0 else nan
    back    = ((last_codes != older) & (flat == older)).sum().to(double) / count if last_codes is not None and older is not None and last_codes.shape == flat.shape == older.shape and count > 0 else nan
    keep    = flat.to(torch.int32 if info.code_total <= 2**31 else torch.int64).clone()
    return torch.stack([near, changed, back]), (keep, last_codes)
