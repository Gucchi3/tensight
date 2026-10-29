# @file quaver_adapter.py
# @brief QuaverAdapter: reads the quantizers and the quantized values of quaver (C-04, ADR-004)
# @author Kohei Yamaguchi, Claude

from torch  import nn
from typing import Any, cast

import sys
import torch

from ..types import QuantInfo, QuantizerSite, QuantPair, QuantValue, RoundMode



CODE_BITS = 24



class QuaverAdapter:
    '''
    #* Read quaver: the calls of its quantizers, and the "(tensor, QuantParams)" values of its layers (FR-012 to FR-015).
    #? registry_adapters() tries this one before TorchAdapter.
    #! tensight does not import quaver. It uses the quaver that the user's script has loaded, as "quaver" or under a name that ends with ".quaver" such as "utils.quaver" (CR-009).
    '''
    
    name = "quaver"
    
    def find_quantizers(self, model: nn.Module) -> list[QuantizerSite]:
        '''
        #* Find every quaver Quantizer, and the calls of it that carry the quantization (03 U-030).
        '''
        
        # Make the sites from the name of the attribute that holds the quantizer
        sites = list[QuantizerSite]()
        for name, module in model.named_modules():
            quaver         = self._package(module)
            if quaver is None or not isinstance(module, quaver.Quantizer):
                continue
            owner, _, attr = name.rpartition(".")
            if attr == "weight_quantizer":
                layer  = model.get_submodule(owner) if owner else model
                param  = (owner + ".weight" if owner else "weight") if isinstance(getattr(layer, "weight", None), nn.Parameter) else None
                sites += [QuantizerSite(module, "weight", owner, param, "forward"), QuantizerSite(module, "weight", owner, param, "encode")]
            elif attr == "output_quantizer":
                sites.append(QuantizerSite(module, "output", owner, None, "measure"))
            else:
                sites.append(QuantizerSite(module, "output", name, None, "forward"))
        return sites
    
    def read_value(self, value: object) -> QuantValue | None:
        '''
        #* Read a "(tensor, QuantParams)" value of a quaver layer.
        #! In the integer mode the tensor holds the device codes: they are counted as they are, the real values are made in float64, and the codes are kept as the source (CR-010).
        '''
        
        # Handle only the pairs of a tensor and a QuantParams of quaver
        if not isinstance(value, tuple) or len(value) != 2 or not isinstance(value[0], torch.Tensor):
            return None
        quaver = self._package(value[1])
        if quaver is None or not isinstance(value[1], quaver.QuantParams):
            return None
        
        # Name the two parts
        tensor, params = value
        
        # Training mode: the tensor holds real values
        if tensor.is_floating_point():
            return QuantValue(tensor, self._info(quaver, params, tensor, None, "nearest"))
        
        # Integer mode: the tensor holds codes; real value = (code - zero point) x code scale
        codes  = tensor.to(torch.int64)
        values = (codes.double() - params.zero_point.double()) * params.code_scale.double()
        return QuantValue(values, self._info(quaver, params, None, codes, "nearest"), tensor)
    
    def read_quantizer(self, site: QuantizerSite, args: tuple[Any, ...], result: object) -> QuantPair | None:
        '''
        #* Read one call of a quantizer: forward (training), encode (integer mode), or measure (outputs).
        '''
        
        # A turned off quantizer returns x as it is
        quaver = self._package(site.module)
        if quaver is None or not getattr(site.module, "enabled", True):
            return None
        
        # Read the rounding of the quantizer, and the values before it
        mode: RoundMode = "floor" if getattr(site.module, "round_mode", None) == "floor" else "nearest"
        before          = args[0]
        
        # forward: fake-quantized values and their parameters
        if site.method == "forward":
            after, params = cast(tuple[torch.Tensor, Any], result)
            return QuantPair(before, after, self._info(quaver, params, after, None, mode))
        
        # encode: the device codes; the real values are made in float64
        if site.method == "encode":
            codes, params = cast(tuple[torch.Tensor, Any], result)
            
            # Real value = (code - zero point) x code scale, in float64
            codes = codes.to(torch.int64)
            after = (codes.double() - params.zero_point.double()) * params.code_scale.double()
            return QuantPair(before, after, self._info(quaver, params, None, codes, mode))
        
        # measure: only the parameters; the quantized values come later as the output of the layer
        if site.method == "measure":
            return QuantPair(before, None, self._info(quaver, result, None, None, mode))
        return None
    
    def _package(self, obj: object) -> Any:
        '''
        #* Return the quaver package that the class of obj comes from, or None (03 U-030, CR-009).
        #! The module name of the class is cut at each "." from the end, until a loaded module whose last part is "quaver" and that has Quantizer and QuantParams is found.
        '''
        
        parts = str(getattr(type(obj), "__module__", "")).split(".")
        for end in range(len(parts), 0, -1):
            module = sys.modules.get(".".join(parts[:end])) if parts[end - 1] == "quaver" else None
            if module is not None and hasattr(module, "Quantizer") and hasattr(module, "QuantParams"):
                return module
        return None
    
    def _info(self, quaver: Any, params: Any, values: torch.Tensor | None, codes: torch.Tensor | None, mode: RoundMode) -> QuantInfo:
        '''
        #* Make the quantization info from quaver's QuantParams (03 U-030).
        #! Codes are made from float values only when bits <= 24: float32 cannot hold larger integers exactly.
        '''
        
        # Read the format and the scales; real value = (code - zero point) x scale x step_min. FP4 values with UE4M3 block scales are NVFP4 (BUG-004)
        fmt        = params.number_format
        name       = "nvfp4" if params.block is not None and fmt.name == "fp4" and params.block.scale_format.name == "ue4m3" else fmt.name
        scale      = params.code_scale.double() / fmt.step_min
        zero_point = params.zero_point.double() if fmt.kind == "int" else None
        
        # Use the given codes, or make them from the values; FP codes are counted as bit patterns
        grid = codes if codes is not None else quaver.params_codes(values, params) if values is not None and fmt.bits <= CODE_BITS else None
        if grid is not None and fmt.kind == "fp":
            grid = quaver.formats.fp_encode(grid.double() * fmt.step_min, fmt)
        
        # Make the levels in code units: INT codes, or FP values
        total       = 2**fmt.bits
        low         = int(fmt.min_value) if fmt.kind == "int" else 0
        levels      = None
        if total <= 4096:
            levels = torch.arange(low, int(fmt.max_value) + 1, dtype=torch.float64) if fmt.kind == "int" else quaver.formats.fp_values(fmt).double()
        granularity = "group" if params.block is not None else "tensor" if scale.numel() == 1 else "channel"
        return QuantInfo(name, fmt.kind, fmt.bits, scale, zero_point, granularity, grid, total, levels, low, mode)
