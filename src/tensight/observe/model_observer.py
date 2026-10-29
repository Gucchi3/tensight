# @file model_observer.py
# @brief ModelObserver: the hooks on one model that hand the tensors of recorded steps to StepCollector (C-02)

from collections.abc   import Sequence
from functools         import partial
from torch             import nn
from torch.utils.hooks import RemovableHandle
from typing            import Any

import torch

from ..core    import message_warn
from ..stats   import StepCollector
from ..types   import Adapter, MapPattern, PointInfo, QuantizerSite, QuantValue, RecordKind, RunSettings
from .capture  import capture_active
from .glob     import glob_compile
from .layout   import layout_choose
from .walk     import walk_value



class ModelObserver:
    '''
    #* Put hooks on the observation points of one model, and hand their tensors to StepCollector in recorded steps (IF-01 to IF-04).
    #? watch_model() makes one for each model. RunState tells it whether the step is recorded (IF-02).
    #! In steps that are not recorded, every hook only reads self.recording and returns (NFR-001). The hooks never change a tensor (NFR-010).
    '''
    
    names    : list[str]
    recording: bool
    step     : int
    
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer | None, settings: RunSettings, collector: StepCollector, adapters: Sequence[Adapter]) -> None:
        '''
        #* Find the layers and the quantizers to observe, and put the hooks on them (03 U-017).
        '''
        
        # Keep the state of the hooks
        self.model     = model
        self.settings  = settings
        self.collector = collector
        self.adapters  = list(adapters)
        self.kinds     = set(settings.kinds)
        self.names     = []
        self.recording = False
        self.step      = 0
        self.handles   = list[RemovableHandle]()
        self.wrapped   = list[tuple[nn.Module, str, object | None]]()
        self.specs     = dict[str, MapPattern | None]()
        self.grad_seen = set[str]()
        self.updates   = list[tuple[str, str, nn.Module, torch.Tensor]]()
        self.snapshots = list[tuple[str, str, nn.Module, torch.Tensor, torch.Tensor]]()
        
        # Find the quantizers; they and their modules do not count as children (02a 4章)
        sites      = list[tuple[Adapter, QuantizerSite]]()
        for adapter in self.adapters:
            try:
                sites += [(adapter, site) for site in adapter.find_quantizers(model)]
            except Exception as error:
                message_warn("MSG-003", adapter=adapter.name, error=error)
        quantizers = {id(part) for _, site in sites for part in site.module.modules()}
        
        # Choose the layers: the leaf layers, or the layers that the patterns match (the model itself is named "model")
        modules = [(name or "model", module) for name, module in model.named_modules()]
        if settings.points is None:
            layers = [(name, module) for name, module in modules if id(module) not in quantizers and all(id(child) in quantizers for child in module.children())]
        else:
            matched = set[str]()
            for pattern in settings.points:
                found = [name for name, _ in modules if glob_compile(pattern).fullmatch(name)]
                if not found:
                    message_warn("MSG-001", pattern=pattern)
                matched.update(found)
            layers  = [(name, module) for name, module in modules if name in matched]
        
        # Put the hooks on each layer and on its direct weights and biases
        observed = set[str]()
        seen     = set[int]()
        for name, module in layers:
            observed.add(name)
            values      = list[tuple[str, torch.Tensor]]()
            grads       = list[str]()
            updates     = list[str]()
            for param_name, param in module.named_parameters(recurse=False):
                if id(param) in seen:
                    continue
                seen.add(id(param))
                full   = param_name if name == "model" else name + "." + param_name
                weight = param_name.endswith("weight")
                bias   = param_name.endswith("bias")
                if (weight and "weight" in self.kinds) or (bias and "bias" in self.kinds):
                    values.append((full, param))
                if (weight or bias) and "grad" in self.kinds and param.requires_grad:
                    self.handles.append(param.register_post_accumulate_grad_hook(partial(self.hook_grad, name, full, module)))
                    grads.append(full + ":grad")
                if (weight or bias) and "update" in self.kinds:
                    self.updates.append((name, full, module, param))
                    updates.append(full + ":update")
            if "input" in self.kinds or values:
                self.handles.append(module.register_forward_pre_hook(partial(self.hook_pre, name, values)))
            if "output" in self.kinds or "output_grad" in self.kinds:
                self.handles.append(module.register_forward_hook(partial(self.hook_post, name)))
            self.names += [name + ":" + kind for kind in ("input", "output", "output_grad") if kind in self.kinds]
            self.names += [full + ":value" for full, _ in values] + grads + updates
        
        # Put the hooks on the quantizers whose point is observed; methods other than forward are wrapped (IF-04)
        for adapter, site in sites:
            owner = site.owner or "model"
            if owner not in observed:
                continue
            if site.role == "output" and "output" in self.kinds:
                point = owner + ":output"
            elif site.role == "input" and "input" in self.kinds:
                point = owner + ":input"
            elif site.role == "weight" and "weight" in self.kinds:
                point = site.param + ":value" if site.param else None
            else:
                continue
            if site.method == "forward":
                self.handles.append(site.module.register_forward_hook(partial(self.hook_site, adapter, site, point), prepend=True))
                continue
            self.wrapped.append((site.module, site.method, site.module.__dict__.get(site.method)))
            setattr(site.module, site.method, partial(self.call_wrapped, getattr(site.module, site.method), adapter, site, point))
        
        # Put the hooks on the optimizer for the updates (FR-028)
        if "update" in self.kinds:
            if optimizer is None:
                message_warn("MSG-008")
            else:
                self.handles.append(optimizer.register_step_pre_hook(self.hook_step_pre))
                self.handles.append(optimizer.register_step_post_hook(self.hook_step_post))
        return None
    
    def set_recording(self, recording: bool, step: int) -> None:
        '''
        #* Say whether the step that runs now is recorded (IF-02).
        '''
        
        self.recording = recording
        self.step      = step
        self.grad_seen.clear()
        self.snapshots.clear()
        return None
    
    def detach(self) -> None:
        '''
        #* Take off every hook, and put the wrapped methods back.
        '''
        
        for handle in self.handles:
            handle.remove()
        for module, method, original in self.wrapped:
            if original is None:
                delattr(module, method)
            else:
                setattr(module, method, original)
        self.handles.clear()
        self.wrapped.clear()
        return None
    
    def hook_pre(self, name: str, values: list[tuple[str, torch.Tensor]], module: nn.Module, args: tuple[Any, ...]) -> None:
        '''
        #* Before the forward of a layer: record its input and the values of its weights and biases.
        '''
        
        if capture_active() or not self.recording:
            return None
        with torch.no_grad():
            if "input" in self.kinds and args:
                for leaf in walk_value(args[0] if len(args) == 1 else args, name + ":input", self.adapters):
                    self._add(name + ":input" + leaf.suffix, name, "input", name, module, leaf.tensor, leaf.quant)
            for full, param in values:
                self._add(full + ":value", full, "value", name, module, param.detach(), None)
        return None
    
    def hook_post(self, name: str, module: nn.Module, args: tuple[Any, ...], output: object) -> None:
        '''
        #* After the forward of a layer: record its output, and catch the gradients of the output in the backward.
        '''
        
        if capture_active() or not self.recording:
            return None
        try:
            leaves = walk_value(output, name + ":output", self.adapters)
            if "output" in self.kinds:
                with torch.no_grad():
                    for leaf in leaves:
                        self._add(name + ":output" + leaf.suffix, name, "output", name, module, leaf.tensor, leaf.quant)
            if "output_grad" in self.kinds:
                for leaf in leaves:
                    point = name + ":output_grad" + leaf.suffix
                    if leaf.tensor.requires_grad and point not in self.grad_seen:
                        self.grad_seen.add(point)
                        leaf.tensor.register_hook(partial(self.hook_output_grad, point, name, module))
        except Exception as error:
            message_warn("MSG-002", point=name + ":output", error=error)
        return None
    
    def hook_output_grad(self, point: str, name: str, module: nn.Module, grad: torch.Tensor) -> None:
        '''
        #* Record the gradient of an output when the backward reaches it; the gradient is not changed.
        '''
        
        if capture_active() or not self.recording:
            return None
        with torch.no_grad():
            self._add(point, name, "output_grad", name, module, grad, None)
        return None
    
    def hook_grad(self, name: str, full: str, module: nn.Module, param: torch.Tensor) -> None:
        '''
        #* Record the gradient of a parameter after the backward has added it; the last one of the step is kept (FR-005).
        '''
        
        if capture_active() or not self.recording or param.grad is None:
            return None
        with torch.no_grad():
            self._add(full + ":grad", full, "grad", name, module, param.grad, None, replace=True)
        return None
    
    def hook_site(self, adapter: Adapter, site: QuantizerSite, point: str | None, module: nn.Module, args: tuple[Any, ...], result: object) -> None:
        '''
        #* Read one call of a quantizer with its reader, and hand the pair to StepCollector (IF-04).
        '''
        
        # Read the call
        if capture_active() or not self.recording:
            return None
        try:
            with torch.no_grad():
                pair = adapter.read_quantizer(site, args, result)
        except Exception as error:
            message_warn("MSG-003", adapter=adapter.name, error=error)
            return None
        if pair is None:
            return None
        
        # A weight quantizer without its parameter name: find the parameter that it received
        owner = site.owner or "model"
        if point is None:
            layer = self.model.get_submodule(site.owner) if site.owner else self.model
            found = next((param_name for param_name, param in layer.named_parameters(recurse=False) if args and param is args[0]), None)
            if found is None:
                return None
            point = (found if owner == "model" else owner + "." + found) + ":value"
        
        # Hand the pair to the point that the quantizer belongs to
        kind: RecordKind = "value" if site.role == "weight" else site.role
        self.collector.add_pair(PointInfo(point, point.rpartition(":")[0], kind, owner), pair)
        return None
    
    def call_wrapped(self, bound: Any, adapter: Adapter, site: QuantizerSite, point: str | None, *args: Any, **kwargs: Any) -> Any:
        '''
        #* Call the wrapped method of a quantizer as it is, then read the call (ADR-004).
        #! The result goes back unchanged.
        '''
        
        result = bound(*args, **kwargs)
        self.hook_site(adapter, site, point, site.module, args, result)
        return result
    
    def hook_step_pre(self, optimizer: torch.optim.Optimizer, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        '''
        #* Before the optimizer step: copy the parameters whose updates are recorded (FR-028).
        '''
        
        if capture_active() or not self.recording:
            return None
        with torch.no_grad():
            self.snapshots = [(name, full, module, param, param.detach().clone()) for name, full, module, param in self.updates if param.grad is not None]
        return None
    
    def hook_step_post(self, optimizer: torch.optim.Optimizer, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        '''
        #* After the optimizer step: record the update (now - before) and its ratio (FR-028).
        '''
        
        if capture_active() or not self.recording:
            return None
        with torch.no_grad():
            for name, full, module, param, before in self.snapshots:
                self._add(full + ":update", full, "update", name, module, param.detach() - before, None, ratio_base=before)
        self.snapshots = []
        return None
    
    def _add(self, point: str, target: str, kind: RecordKind, layer: str, module: nn.Module, tensor: torch.Tensor, quant: QuantValue | None, replace: bool = False, ratio_base: torch.Tensor | None = None) -> None:
        '''
        #* Choose the map pattern of a point once, and hand the tensor to StepCollector (IF-03).
        '''
        
        try:
            if point not in self.specs:
                spec = next((found for key, found in self.settings.maps if glob_compile(key).fullmatch(point)), None)
                if spec == "auto":
                    pattern, warn     = layout_choose(module, kind, tensor.ndim)
                    if warn:
                        message_warn("MSG-013", point=point, layer=type(module).__name__, pattern=pattern.text if pattern is not None else "なし")
                    self.specs[point] = pattern
                else:
                    self.specs[point] = spec if isinstance(spec, MapPattern) else None
            self.collector.add_tensor(PointInfo(point, target, kind, layer), tensor, self.specs[point], quant, replace, ratio_base)
        except Exception as error:
            message_warn("MSG-002", point=point, error=error)
        return None
