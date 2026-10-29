# @file step_collector.py
# @brief StepCollector: gathers the summaries of one step, and hands them to the writer at its end (C-03)
# @author Kohei Yamaguchi, Claude

import dataclasses
import math
import torch

from ..core    import message_warn
from ..types   import MapPattern, PointInfo, QuantInfo, QuantPair, QuantValue, RunSettings, StepRecord
from ..writer  import RecordWriter
from .pattern  import pattern_plan
from .quant    import quant_boundary, quant_summary
from .summary  import summary_move, summary_parts, summary_report, summary_tensor
from .types    import CollectorEntry, PendingReport
from .values   import values_prepare, values_problem



class StepCollector:
    '''
    #* Gather the summaries of the points in one step, and hand them to the writer at the end of the step (IF-03, IF-04, IF-05).
    #? RunState makes one for each run. ModelObserver and note_tensor() add tensors, and RunState calls finish().
    #! add_tensor() and add_pair() never wait for the GPU. finish() waits once for each device (NFR-002).
    '''
    
    def __init__(self, settings: RunSettings, writer: RecordWriter) -> None:
        '''
        #* Keep the settings and the writer, with empty entries.
        '''
        
        self.settings = settings
        self.writer   = writer
        self.step     = 0
        self.entries  = dict[str, CollectorEntry]()
        self.history  = dict[str, tuple[torch.Tensor | None, torch.Tensor | None]]()
        self.index    = dict[tuple[str, str], tuple[tuple[int, ...], torch.Tensor]]()
        self.offsets  = dict[tuple[str, str, tuple[int, ...]], torch.Tensor]()
        return None
    
    def begin(self, step: int) -> None:
        '''
        #* Start gathering a step, forgetting what was gathered for another step.
        '''
        
        if step != self.step:
            self.entries.clear()
            self.step = step
        return None
    
    def add_tensor(self, point: PointInfo, tensor: torch.Tensor, pattern: MapPattern | None, quant: QuantValue | None = None, replace: bool = False, ratio_base: torch.Tensor | None = None) -> None:
        '''
        #* Summarize one tensor of a point on its device (IF-03).
        #! The first tensor of a point in a step is kept; replace=True (gradients) keeps the last one (02a 4章).
        '''
        
        # Keep the first value of the step, unless this one replaces it
        entry = self.entries.get(point.name)
        if entry is not None and entry.pending is not None and not replace:
            return None
        try:
            with torch.no_grad():
                
                # Skip the tensors that cannot be recorded (FR-039)
                problem = values_problem(tensor)
                if problem is not None:
                    message_warn("MSG-012", point=point.name, reason=problem)
                    return None
                values  = values_prepare(quant.values if quant is not None else tensor)
                
                # Fit the map pattern to the shape (FR-035)
                plan = pattern_plan(pattern, tuple(values.shape)) if pattern is not None else None
                if pattern is not None and plan is None:
                    message_warn("MSG-009", point=point.name, ndim=values.ndim, pattern=pattern.text)
                
                # Summarize the tensor, with the followed elements (FR-032) and the copies to save
                chosen, index = self._positions(point, values)
                keep_compare  = self.step in self.settings.compare_steps
                keep_full     = self.step in self.settings.save_full.get(point.name, frozenset())
                pending       = summary_tensor(values, tensor, self.settings.bins, self.settings.sample_size, index, chosen, plan, pattern, keep_compare, keep_full)
                entry         = entry if entry is not None else CollectorEntry(point)
                entry.pending = pending
                
                # Summarize the quantization; a pair that came first (after is None) gives the values before the quantizer
                if entry.quant is None and (quant is not None or entry.wait is not None):
                    found = None
                    if entry.wait is not None:
                        found = QuantPair(entry.wait.before, values, dataclasses.replace(entry.wait.info, codes=quant.info.codes) if quant is not None else entry.wait.info)
                    elif quant is not None:
                        found = quant
                    if found is not None:
                        entry.quant     = quant_summary(found, values)
                        entry.pair_info = found.info
                        entry.wait      = None
                        if point.kind == "value" and isinstance(found, QuantPair) and found.info.codes is not None:
                            entry.boundary, self.history[point.name] = quant_boundary(found.before, found.info, found.info.codes, self.history.get(point.name, (None, None)), self.settings.boundary_width)
                
                # Read the scales and the zero points of the followed elements
                if entry.pair_info is not None and pending.elements is not None:
                    entry.scales, entry.zeros = self._element_scales(point, pending, entry.pair_info)
                
                # The update ratio: |update| / |weight before the update| (FR-028)
                if ratio_base is not None:
                    norm        = ratio_base.detach().to(torch.float64).norm()
                    entry.ratio = torch.where(norm > 0, tensor.detach().to(torch.float64).norm() / norm, torch.full((), math.nan, dtype=torch.float64, device=norm.device))
                self.entries[point.name] = entry
        except Exception as error:
            self.entries.pop(point.name, None)
            message_warn("MSG-002", point=point.name, error=error)
        return None
    
    def add_pair(self, point: PointInfo, pair: QuantPair) -> None:
        '''
        #* Summarize one call of a quantizer for a point (IF-04).
        #! A pair whose after is None waits for the values of the point, and is used only when it comes before them.
        '''
        
        # Keep the first call of the step
        entry = self.entries.get(point.name)
        if entry is not None and entry.quant is not None:
            return None
        entry = entry if entry is not None else CollectorEntry(point)
        try:
            with torch.no_grad():
                
                # Wait for the values of the point when the quantized values come later
                if pair.after is None:
                    if entry.pending is None:
                        entry.wait               = pair
                        self.entries[point.name] = entry
                    return None
                
                # Summarize the pair, and the moves near the boundaries of a parameter (FR-031)
                entry.quant              = quant_summary(pair, pair.after)
                entry.pair_info          = pair.info
                if point.kind == "value" and pair.info.codes is not None:
                    entry.boundary, self.history[point.name] = quant_boundary(pair.before, pair.info, pair.info.codes, self.history.get(point.name, (None, None)), self.settings.boundary_width)
                if entry.pending is not None and entry.pending.elements is not None:
                    entry.scales, entry.zeros = self._element_scales(point, entry.pending, pair.info)
                self.entries[point.name] = entry
        except Exception as error:
            message_warn("MSG-003", adapter="不明", error=error)
        return None
    
    def finish(self, step: int) -> None:
        '''
        #* Move the summaries of the step to the CPU in one copy for each device, and hand them to the writer (IF-05).
        '''
        
        # Collect the device tensors of each point, grouped by device
        plans   = list[tuple[CollectorEntry, PendingReport, str, int, int, int]]()
        devices = dict[str, list[torch.Tensor]]()
        for entry in self.entries.values():
            pending = entry.pending
            if pending is None:
                continue
            parts   = summary_parts(pending, entry.quant)
            extras  = [part for part in (pending.elements, entry.scales, entry.zeros, entry.boundary, entry.ratio) if part is not None]
            group   = devices.setdefault(str(pending.numbers.device), [])
            plans.append((entry, pending, str(pending.numbers.device), len(group), len(parts), len(extras)))
            group  += parts + extras
        if not plans:
            self.entries.clear()
            return None
        try:
            with torch.no_grad():
                
                # Move each device's tensors in one copy; this is where the step waits for the GPU
                moved = {device: summary_move(group) for device, group in devices.items()}
                
                # Make the records, in the order the points came
                records = list[StepRecord]()
                for entry, pending, device, first, count, extra in plans:
                    cpu      = moved[device][first:first + count + extra]
                    report   = summary_report(pending, entry.quant, cpu[:count])
                    extras   = iter(cpu[count:])
                    values   = next(extras).tolist() if pending.elements is not None else None
                    scales   = next(extras).tolist() if entry.scales is not None else None
                    zeros    = next(extras).tolist() if entry.zeros is not None else None
                    bound    = next(extras).tolist() if entry.boundary is not None else None
                    ratio    = float(next(extras).item()) if entry.ratio is not None else None
                    elements = {position: (values[i], scales[i] if scales is not None else None, zeros[i] if zeros is not None else None) for i, position in enumerate(pending.element_index)} if values is not None else None
                    records.append(StepRecord(entry.point, report, elements, (bound[0], bound[1], bound[2]) if bound is not None else None, ratio))
                
                # Move the copies to save, and hand everything to the writer
                compares = [(entry.point, pending.compare.to("cpu")) for entry, pending, _, _, _, _ in plans if pending.compare is not None]
                fulls    = [(entry.point, pending.full.to("cpu")) for entry, pending, _, _, _, _ in plans if pending.full is not None]
                self.writer.write_step(step, records, compares, fulls)
        except Exception as error:
            message_warn("MSG-002", point="このステップのすべての観測点", error=error)
        self.entries.clear()
        return None
    
    def clear(self) -> None:
        '''
        #* Forget what was gathered in this step; the code history and the cached positions stay.
        '''
        
        self.entries.clear()
        return None
    
    def _positions(self, point: PointInfo, values: torch.Tensor) -> tuple[tuple[int, ...], torch.Tensor | None]:
        '''
        #* Return the followed elements of a point and their positions on the device, made once for each point (FR-032).
        '''
        
        # Choose the elements: the given ones, or the first sample_size values of a parameter
        tracked = self.settings.track_elements
        if tracked is not None and point.name in tracked:
            chosen = tracked[point.name]
        elif tracked is None and point.kind == "value":
            chosen = tuple(range(self.settings.sample_size))
        else:
            return (), None
        chosen  = tuple(position for position in chosen if position < values.numel())
        if not chosen:
            return (), None
        
        # Put the positions on the device once, and use them again
        key    = (point.name, str(values.device))
        cached = self.index.get(key)
        if cached is None or cached[0] != chosen:
            cached          = (chosen, torch.tensor(chosen, dtype=torch.int64, device=values.device))
            self.index[key] = cached
        return cached
    
    def _element_scales(self, point: PointInfo, pending: PendingReport, info: QuantInfo) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        '''
        #* Return the scale and the zero point of each followed element, from the scales broadcast to the shape (FR-032).
        #! A scale that does not broadcast to the shape gives None.
        '''
        
        found = list[torch.Tensor | None]()
        for part in (info.scale, info.zero_point):
            if part is None:
                found.append(None)
                continue
            
            # Find the offset of each element in the broadcast scales, once for each point and shape
            base = part.detach().contiguous()
            key  = (point.name, str(base.device), tuple(base.shape))
            if key not in self.offsets:
                try:
                    strides = torch.broadcast_to(base, pending.shape).stride()
                except RuntimeError:
                    found.append(None)
                    continue
                offsets           = list[int]()
                for position in pending.element_index:
                    offset = 0
                    rest   = position
                    for size, stride in zip(reversed(pending.shape), reversed(strides)):
                        offset += (rest % size) * stride
                        rest  //= size
                    offsets.append(offset)
                self.offsets[key] = torch.tensor(offsets, dtype=torch.int64, device=base.device)
            found.append(base.reshape(-1).index_select(0, self.offsets[key]).to(torch.float64))
        return found[0], found[1]
