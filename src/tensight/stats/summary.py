# @file summary.py
# @brief Statistics, distribution, first values, element values and the map of one tensor (FR-008 to FR-010, FR-032, FR-035)

import math
import torch

from ..types import CodeUsage, Histogram, MapPattern, MapReport, QuantReport, TensorReport, ValueSummary
from .map    import map_build
from .types  import MapPlan, PendingQuant, PendingReport



SUMMARY_CHUNK = 2**22



def summary_tensor(values: torch.Tensor, source: torch.Tensor, bins: int, sample_size: int, elements: torch.Tensor | None, element_index: tuple[int, ...], plan: MapPlan | None, pattern: MapPattern | None,
                   keep_compare: bool, keep_full: bool) -> PendingReport:
    '''
    #* Compute the summary of one tensor on its device, without waiting for the GPU (03 U-023).
    #? Called from StepCollector and inspect_tensor(). summary_parts() and summary_report() finish it on the CPU.
    #! values come from values_prepare(). elements holds positions from 0 up to numel - 1 on the device of values.
    '''
    
    # Keep the running numbers on the device
    flat   = values.reshape(-1)
    numel  = flat.numel()
    device = flat.device
    double = torch.float64
    low    = torch.full((), math.inf, dtype=double, device=device)
    high   = torch.full((), -math.inf, dtype=double, device=device)
    total  = torch.zeros((), dtype=double, device=device)
    square = torch.zeros((), dtype=double, device=device)
    spread = torch.zeros((), dtype=double, device=device)
    count  = torch.zeros((), dtype=torch.int64, device=device)
    zero   = torch.zeros((), dtype=torch.int64, device=device)
    nan    = torch.zeros((), dtype=torch.int64, device=device)
    inf    = torch.zeros((), dtype=torch.int64, device=device)
    counts = torch.zeros(bins + 1, dtype=torch.int64, device=device)
    
    # First pass: counts, the smallest and largest finite values, the sum, and the sum of squares
    for start in range(0, numel, SUMMARY_CHUNK):
        part    = flat[start:start + SUMMARY_CHUNK]
        finite  = torch.isfinite(part)
        kept    = torch.where(finite, part, 0).to(double)
        count  += finite.sum()
        nan    += torch.isnan(part).sum()
        inf    += torch.isinf(part).sum()
        zero   += (part == 0).sum()
        low     = torch.minimum(low, torch.where(finite, part, math.inf).amin().to(double))
        high    = torch.maximum(high, torch.where(finite, part, -math.inf).amax().to(double))
        total  += kept.sum()
        square += (kept * kept).sum()
    
    # Second pass: the spread around the mean, and the distribution between the smallest and the largest value (FR-009)
    mean  = total / count.clamp(min=1)
    width = high - low
    scale = torch.where(width > 0, bins / width, torch.zeros((), dtype=double, device=device))
    for start in range(0, numel, SUMMARY_CHUNK):
        part    = flat[start:start + SUMMARY_CHUNK]
        finite  = torch.isfinite(part)
        moved   = torch.where(finite, part.to(double) - mean, 0)
        spread += (moved * moved).sum()
        index   = torch.where(finite, ((part.to(double) - low) * scale).floor().clamp(0, bins - 1), bins).to(torch.int64)
        counts.scatter_add_(0, index, torch.ones(1, dtype=torch.int64, device=device).expand(index.numel()))
    
    # Keep the first values, the followed elements, the map, and the copies to save
    numbers = torch.stack([low, high, total, spread, count.to(double), zero.to(double), nan.to(double), inf.to(double), square])
    sample  = flat[:sample_size].to(double).clone()
    chosen  = flat.index_select(0, elements).to(double) if elements is not None else None
    cells   = map_build(values, plan) if plan is not None and numel > 0 else None
    compare = values.detach().to(torch.float32, copy=True) if keep_compare else None
    full    = source.detach().clone() if keep_full else None
    return PendingReport(str(source.dtype).removeprefix("torch."), tuple(source.shape), str(source.device), numel, bins, numbers, counts, sample, chosen, element_index, cells, plan if cells is not None else None, pattern, compare, full)


def summary_parts(pending: PendingReport, quant: PendingQuant | None) -> list[torch.Tensor]:
    '''
    #* List the device tensors of a summary, in the order that summary_report() reads them (03 U-024).
    #! The followed elements are not here: StepCollector moves them with its own tensors. The levels stay where the reader made them (on the CPU), so they are not here either.
    '''
    
    parts = [pending.numbers, pending.counts, pending.sample]
    if pending.map is not None:
        parts.append(pending.map)
    if quant is not None:
        parts += [part for part in (quant.scale, quant.zero_point, quant.counts, quant.used, quant.clip, quant.errors) if part is not None]
    return parts


def summary_move(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    '''
    #* Move tensors of one device to the CPU in one copy, as float64, and give them back with their shapes.
    #? StepCollector and inspect_tensor() use this; it is the one wait for the GPU in a recorded step (NFR-002).
    '''
    
    # Join them into one row, and move it once
    if not tensors:
        return []
    moved = torch.cat([tensor.reshape(-1).to(torch.float64) for tensor in tensors]).to("cpu")
    
    # Cut the row back into the tensors
    parts = list[torch.Tensor]()
    start = 0
    for tensor in tensors:
        parts.append(moved[start:start + tensor.numel()].reshape(tensor.shape))
        start += tensor.numel()
    return parts


def summary_report(pending: PendingReport, quant: PendingQuant | None, parts: list[torch.Tensor]) -> TensorReport:
    '''
    #* Make the TensorReport from the CPU copies of the parts, in the order of summary_parts() (03 U-024).
    '''
    
    # Read the parts in the order of summary_parts()
    reader  = iter(parts)
    numbers = next(reader).tolist()
    counts  = next(reader)
    sample  = tuple(next(reader).tolist())
    
    # Name the numbers in the order of PendingReport
    low, high, total, spread, count, zero, nan, inf, square = numbers
    
    # Without finite values, the statistics and the distribution are None (FR-008, FR-009)
    found = count > 0
    hist  = Histogram(low, high, tuple(int(n) for n in counts[:pending.bins].tolist())) if found else Histogram(None, None, ())
    
    # Read the map (FR-035)
    map_ = None
    if pending.map is not None and pending.plan is not None and pending.pattern is not None:
        values = next(reader).to(torch.float32)
        preset = pending.pattern.preset or "custom"
        map_   = MapReport(preset, pending.pattern.text, values.shape[0], values.shape[1], pending.plan.row_size, pending.plan.col_size, pending.plan.row_names, pending.plan.col_names, float(values.max()) if values.numel() else 0.0, values)
    
    # Read the quantization (FR-012 to FR-015)
    report = None
    if quant is not None:
        scale      = next(reader).tolist()
        zero_point = next(reader).tolist() if quant.zero_point is not None else None
        codes      = next(reader) if quant.counts is not None else None
        used       = int(next(reader).item()) if quant.used is not None else None
        levels     = tuple(quant.levels.detach().to("cpu", torch.float64).tolist()) if quant.levels is not None else None
        clip       = next(reader).tolist() if quant.clip is not None else None
        errors     = next(reader).tolist() if quant.errors is not None else None
        usage      = CodeUsage(quant.code_total, used, quant.code_min, tuple(int(n) for n in codes.tolist()) if codes is not None else None) if used is not None else None
        sqnr       = None
        
        # SQNR: infinite without error, none without signal (FR-015)
        if errors is not None and errors[3] > 0:
            if errors[0] == 0:
                sqnr = math.inf if errors[2] > 0 else None
            elif errors[2] > 0:
                sqnr = 10.0 * math.log10(errors[2] / errors[0])
        report = QuantReport(
            quant.format,
            quant.kind,
            quant.bits,
            ValueSummary(int(scale[0]), scale[1], scale[2], scale[3]),
            ValueSummary(int(zero_point[0]), zero_point[1], zero_point[2], zero_point[3]) if zero_point is not None else None,
            quant.granularity,
            usage,
            levels,
            quant.round_mode,
            clip[0] / clip[1] if clip is not None and clip[1] > 0 else None,
            errors[0] / errors[3] if errors is not None and errors[3] > 0 else None,
            errors[1] if errors is not None and errors[3] > 0 else None,
            sqnr,
        )
    return TensorReport(
        pending.dtype,
        pending.shape,
        pending.device,
        pending.numel,
        low if found else None,
        high if found else None,
        total / count if found else None,
        math.sqrt(spread / count) if found else None,
        math.sqrt(square) if found else None,
        zero / pending.numel if pending.numel else 0.0,
        int(nan),
        int(inf),
        hist,
        sample,
        report,
        map_,
    )
