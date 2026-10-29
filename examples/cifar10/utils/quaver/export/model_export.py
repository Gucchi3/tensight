# @file model_export.py
# @brief Export a model in the integer mode for the device: model.json (layers, settings, formats, and where each data item is) and model.bin (every data item packed, ready for PSRAM)
# @author Kohei Yamaguchi, Claude

##########################################################
'''
Add a device op to the export
1. Write the op with device_define() (formats/device.py), and call it from the integer function of the op. Note its output with params_note() when its parameters are known there.
2. Add "name: {tensor argument or output index: its format argument, or a format name}" to EXPORT_FORMATS, so its codes get their format.
3. Add "name: {tensor argument: role}" to EXPORT_ROLES, so its data items get the same name in every layer (weight, multiplier, ...).
Add a layer class
1. Give it integer_mode, and run its integer mode with device ops. Its settings go to the JSON when their names are in EXPORT_SETTINGS.
2. Write its role names and settings in docs/設計.md (9章, "実機との約束").
'''
##########################################################

from collections.abc import Callable
from pathlib         import Path
from torch           import nn
from torch.fx.node   import Node, map_arg
from typing          import Any, cast

import inspect
import json
import math
import operator
import re
import torch

from ..formats      import build_format
from ..ops          import ACCUMULATOR_FORMAT, WIDE_ACCUMULATOR_FORMAT, accumulate_bits
from ..scales       import params_named
from .model_outputs import ModelOutputs
from .pack          import pack_codes
from .types         import ExportLayout

# The format of the codes of each device op: tensor arguments by name, and outputs by index, give the argument that names their format, or the name of the format itself
# (zero points have the format of their codes, multipliers are 16 bits, and shifts fit int8: -30 to 94)
EXPORT_FORMATS: dict[str, dict[str | int, str]] = {
    "quantize"           : {"zero_point": "number_format", 0: "number_format"},
    "accumulate_conv2d"  : {"q_x": "x_format", "x_zero_point": "x_format", "x_block": "x_scale_format", "q_w": "w_format", "w_zero_point": "w_format", "w_block": "w_scale_format", 0: "out_format"},
    "accumulate_linear"  : {"q_x": "x_format", "x_zero_point": "x_format", "x_block": "x_scale_format", "q_w": "w_format", "w_zero_point": "w_format", "w_block": "w_scale_format", 0: "out_format"},
    "accumulate_matmul"  : {"q_a": "a_format", "a_zero_point": "a_format", "a_block": "a_scale_format", "q_b": "b_format", "b_zero_point": "b_format", "b_block": "b_scale_format", 0: "out_format"},
    "activation"         : {},
    "requantize"         : {"multiplier": "uint16", "shift": "int8", "out_zero_point": "out_format", 0: "out_format"},
    "add"                : {"a_multiplier": "uint16", "a_shift": "int8", "b_multiplier": "uint16", "b_shift": "int8", "out_zero_point": "out_format", 0: "out_format"},
    "pool_avg2d"         : {"q_x": "number_format", "zero_point": "number_format", 0: "number_format"},
    "normalize_l1"       : {"multiplier": "uint16", "shift": "int8", 0: "out_format", 1: "uint21"},
    "normalize_l1_scores": {"multiplier": "uint16", "shift": "int8", "out_zero_point": "out_format", 0: "out_format"},
    "block_quantize"     : {"scale_multiplier": "uint16", "scale_shift": "int8", "value_multiplier": "uint16", "value_shift": "int8", 0: "out_format", 1: "scale_format"},
    "dynamic_scale"      : {},
    "fixed_point"        : {0: "uint16", 1: "int8"},
}

# The role of each constant argument of an op: the name of the data item in its layer (the same in every instance of a class). Others are named "op_argument"
EXPORT_ROLES: dict[str, dict[str, str]] = {
    "quantize"           : {"scale": "scale", "zero_point": "zero_point"},
    "accumulate_conv2d"  : {"x_zero_point": "input_zero_point", "x_block": "input_block_codes", "q_w": "weight", "w_zero_point": "weight_zero_point", "w_block": "weight_block_codes"},
    "accumulate_linear"  : {"x_zero_point": "input_zero_point", "x_block": "input_block_codes", "q_w": "weight", "w_zero_point": "weight_zero_point", "w_block": "weight_block_codes"},
    "accumulate_matmul"  : {"a_zero_point": "a_zero_point", "a_block": "a_block_codes", "q_b": "b", "b_zero_point": "b_zero_point", "b_block": "b_block_codes"},
    "activation"         : {"zero_point": "clamp_min", "six": "clamp_max"},
    "requantize"         : {"zero_point": "accumulator_zero_point", "multiplier": "multiplier", "shift": "shift", "out_zero_point": "output_zero_point"},
    "add"                : {"a_zero_point": "a_zero_point", "a_multiplier": "a_multiplier", "a_shift": "a_shift", "b_zero_point": "b_zero_point", "b_multiplier": "b_multiplier", "b_shift": "b_shift", "out_zero_point": "output_zero_point"},
    "pool_avg2d"         : {"zero_point": "input_zero_point"},
    "normalize_l1"       : {"zero_point": "neq_zero_point", "multiplier": "neq_multiplier", "shift": "neq_shift"},
    "normalize_l1_scores": {"multiplier": "scores_multiplier", "shift": "scores_shift", "out_zero_point": "scores_zero_point"},
    "block_quantize"     : {"zero_point": "block_zero_point", "scale_multiplier": "block_scale_multiplier", "scale_shift": "block_scale_shift", "value_multiplier": "block_value_multiplier", "value_shift": "block_value_shift"},
    "aten.add"           : {"other": "bias"},
}

# The roles of outputs after the first one, by op and index: the layer output that gives them is "output_role"
EXPORT_OUTPUTS: dict[str, dict[int, str]] = {
    "block_quantize": {1: "block_codes"},
    "normalize_l1"  : {1: "peak"},
}

# The settings of a layer that go to the JSON, when the module has them
EXPORT_SETTINGS = ("stride", "padding", "dilation", "groups", "activation", "style", "round_mode", "heads", "key_dim", "value_dim", "neq_scope", "output_block", "kernel_size")

# The names of the layouts (hwc and chw are the names for 2-D), and the names of the spatial axes for 1, 2, and 3 of them
EXPORT_LAYOUTS = {"channels_last": "channels_last", "channels_first": "channels_first", "hwc": "channels_last", "chw": "channels_first"}
EXPORT_SPATIAL = {1: ["k"], 2: ["h", "w"], 3: ["d", "h", "w"]}

# The names of the torch types of the values
EXPORT_TYPES = {
    torch.bool   : "bool"   ,
    torch.uint8  : "uint8"  ,
    torch.int8   : "int8"   ,
    torch.int16  : "int16"  ,
    torch.int32  : "int32"  ,
    torch.int64  : "int64"  ,
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
}



def model_export(model: nn.Module, inputs: tuple[torch.Tensor, ...], folder: str | Path, alignment: int = 4, layout: dict[str, ExportLayout] | None = None, json_values: bool = True) -> dict[str, Any]:
    '''
    #* Export a model in the integer mode to folder/model.json and folder/model.bin for the device, and return the JSON as a dict.
    #? model.json: the layers in order (blocks hold "layers", other layers their "ops"), their inputs and outputs by name with format and shape, their settings, and for every data item its role, format, encoding, shape, layout, offset, and size.
    #? model.bin: every data item packed back to back, each starting at a multiple of alignment bytes. An offset is the address in PSRAM, counted from the start of the file.
    #? torch.export captures forward(*inputs), and every value that does not depend on the inputs is computed here (weight codes, bias, M0 and n, zero points, block codes). The ops inside the layers stay as detail, which the reference reader runs.
    #? layout maps layer paths or class names to "channels_last" (the default, "hwc" for 2-D) or "channels_first" ("chw"). A child layer takes the layout of its block. channels_last stores conv weights as (out, spatial..., in).
    #? When a flatten of channels_last activations feeds a Linear, the columns of its weight are reordered to (spatial..., channel), so the device result is the same.
    #? json_values=True also writes the numbers of every data item in decimal ("values", in the order of model.bin, nested by stored_shape): INT codes as integers, FP codes as their real values (fp4 1.5), floats as floats.
    #! Put the model in the integer mode (switch_integer_mode) and in eval() first. forward() may return tensors, pairs (codes, QuantParams), or a tuple or list of them. Needs numpy (the "export" extra).
    '''
    
    # Check here at the entry: the integer mode, eval(), requantization with a multiplier and a shift, the alignment, and the names of the layouts
    if not all(getattr(module, "integer_mode") for module in model.modules() if hasattr(module, "integer_mode")):
        raise ValueError("Put the model in the integer mode first: switch_integer_mode(model, True).")
    if any(module.training for module in model.modules()):
        raise ValueError("Put the model in eval() first, so that the scales stay fixed and become constants.")
    if any(getattr(module, "style", None) == "exact" for module in model.modules()):
        raise ValueError("The style 'exact' keeps real ratios, but the device needs M0 and n. Use mul16_shift or shift_only.")
    if alignment < 1:
        raise ValueError(f"alignment must be 1 or more, not {alignment}.")
    if any(value not in EXPORT_LAYOUTS for value in (layout or {}).values()):
        raise ValueError(f"Unknown layout in {layout}. Available: {', '.join(EXPORT_LAYOUTS)}")
    
    # Capture the graph without gradients (so no_grad() inside the layers adds nothing). ModelOutputs gives the codes of the pairs that forward() returns, with notes of their parameters
    # torch.export reads the old TF32 flag of cuDNN, which PyTorch 2.14 allows only while conv and RNN are both "tf32", so not after precision_disable_tf32().
    # The flags go back right after. The integer mode computes in float64 and int64 and the capture runs no real math, so TF32 changes nothing here
    wrapped    = ModelOutputs(model)
    cudnn      = [getattr(torch.backends.cudnn, kind) for kind in ("conv", "rnn")]
    precisions = [backend.fp32_precision for backend in cudnn]
    try:
        for backend in cudnn:
            backend.fp32_precision = "tf32"
        with torch.no_grad():
            program = torch.export.export(wrapped, inputs)
    finally:
        for backend, precision in zip(cudnn, precisions):
            backend.fp32_precision = precision
    
    # The values of the parameters, the buffers, and the constants, and the names of the arguments of forward()
    signature = program.graph_signature
    tensors   = {**program.state_dict, **program.constants}
    sources   = {**signature.inputs_to_parameters, **signature.inputs_to_buffers, **signature.inputs_to_lifted_tensor_constants}
    known     = {name: tensors[target] for name, target in sources.items()}
    nodes     = [node for node in program.graph.nodes if node.op == "call_function"]
    arguments = [name for name, parameter in inspect.signature(model.forward).parameters.items() if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)]
    
    # The inputs keep the names of the arguments of forward()
    rename: dict[str, str] = {str(name): str(argument) for name, argument in zip(signature.user_inputs, arguments)} if len(arguments) >= len(signature.user_inputs) else {}
    
    # Fold every node whose inputs are all known: the scales, M0 and n, the codes of the weights and their block scales, and the bias codes
    for node in nodes:
        if all(value.name in known for value in node.all_input_nodes):
            args, kwargs     = map_arg((node.args, node.kwargs), lambda value: known[value.name])
            known[node.name] = cast(Callable[..., Any], node.target)(*cast(tuple, args), **cast(dict, kwargs))
    
    # Notes and casts that keep every value are not ops: each one stands for the value it reads
    notes = {node.name: node for node in nodes if node.target is torch.ops.quaver.note.default}
    alias = {}
    for node in nodes:
        source  = node.all_input_nodes[0] if node.all_input_nodes else None
        before  = source.meta.get("val") if source is not None else None
        after   = node.meta.get("val")
        if source is None or not isinstance(before, torch.Tensor) or not isinstance(after, torch.Tensor) or source.name in known or node.name in known:
            continue
        integer = not before.dtype.is_floating_point and not after.dtype.is_floating_point and torch.bool not in (before.dtype, after.dtype)
        widens  = integer and torch.iinfo(after.dtype).min <= torch.iinfo(before.dtype).min and torch.iinfo(before.dtype).max <= torch.iinfo(after.dtype).max
        keeps   = getattr(node.target, "__name__", "").split(".")[0] in ("to", "_to_copy", "detach", "clone", "alias") and len(node.all_input_nodes) == 1 and (before.dtype == after.dtype or widens) and tuple(before.shape) == tuple(after.shape)
        if node.name in notes or keeps:
            alias[node.name] = alias.get(source.name, rename.get(source.name, source.name))
    
    # The values that the outputs need, walking back from the outputs (known values become data items there)
    results: list[Node] = []
    map_arg(program.graph.output_node().args[0], results.append)
    if any(node.name in known for node in results):
        raise ValueError("An output does not depend on the inputs, so it has no node to export.")
    
    # Walk back over the nodes that are not known (a note or a kept cast leads to the value it reads)
    needed  = set()
    pending = list(results)
    while pending:
        node = pending.pop()
        if node.name in alias:
            pending.append(node.all_input_nodes[0])
        elif node.name not in needed and node.name not in known:
            needed.add(node.name)
            pending.extend(node.all_input_nodes)
    
    # The metadata of every value: the format, and the scale and the zero point when they are known (a value of the graph by name). The inputs say their type
    metadata: dict[str, dict[str, Any]] = {str(rename.get(node.name, node.name)): {"format": EXPORT_TYPES[node.meta["val"].dtype]} for node in program.graph.nodes if node.name in signature.user_inputs}
    
    # One record for each needed op, and the chain of layers (modules with integer_mode, outside in, below the model) that it runs in
    constants    : dict[str, dict] = {}
    biases       : dict[str, str]  = {}
    labels       : dict[str, str]  = {}
    named_outputs: dict[str, str]  = {}
    records      : list[dict]      = []
    for node in nodes:
        
        # A note writes the format, the scale, and the zero point on the value it notes: constants as numbers, and values of the graph by name (a note of a constant folds away)
        if node.name in notes and node.name not in alias:
            continue
        if node.name in notes:
            target                          = alias[node.name]
            _, scale, zero_point, form, key = node.args
            
            # The format (a wide note keeps int32 on a sum that the export wrote as int32: activation_integer() notes the accumulator with its wide format), then the scale and the zero point
            narrowed                                  = form == WIDE_ACCUMULATOR_FORMAT.name and metadata.get(target, {}).get("format") == ACCUMULATOR_FORMAT.name
            metadata.setdefault(target, {})["format"] = ACCUMULATOR_FORMAT.name if narrowed else cast(str, form)
            for name, value in ((key, scale), ("zero_point", zero_point)):
                if isinstance(value, Node) and value.name in known:
                    metadata[target][cast(str, name)] = torch.as_tensor(known[value.name]).tolist()
                elif isinstance(value, Node) and value.name in needed:
                    metadata[target][cast(str, name)] = {"value": alias.get(value.name, value.name)}
            continue
        if node.name not in needed or node.name in alias or node.target is operator.getitem:
            continue
        if not hasattr(node.target, "_schema"):
            raise ValueError(f"Cannot export {node.target} ({node.name}): only torch ops and quaver device ops can be exported.")
        
        # The schema, the format names of the device op, the arguments by name, and the layers
        schema       = getattr(node.target, "_schema")
        domain, name = schema.name.split("::")
        spec         = EXPORT_FORMATS.get(name, {}) if domain == "quaver" else {}
        bound        = {argument.name: node.args[index] if index < len(node.args) else node.kwargs.get(argument.name) for index, argument in enumerate(schema.arguments)}
        stack        = node.meta.get("nn_module_stack", {})
        chain        = [(key, place.removeprefix("model."), type(wrapped.get_submodule(place)).__name__) for key, (place, _) in stack.items() if place.startswith("model.") and hasattr(wrapped.get_submodule(place), "integer_mode")]
        names        = []
        roles        = []
        attributes   = {}
        
        # An accumulator says int32 when its worst case fits 32 bits (accumulate_bits(), with the largest |code| of the bias added right after it), and the bias takes the format of its accumulator
        if name.startswith("accumulate_") and domain == "quaver":
            adds  = [user for user in node.users if user.target is torch.ops.aten.add.Tensor and isinstance(user.args[1], Node) and user.args[1].name in known]
            peak  = max((int(torch.as_tensor(known[cast(Node, user.args[1]).name]).abs().max()) for user in adds), default=0)
            sides = [params_named(torch.zeros(()), cast(str, bound[f"{side}_format"]), torch.zeros(()) if bound[f"{side}_scale_format"] else None, cast(str, bound[f"{side}_scale_format"])) for side in ("ab" if name == "accumulate_matmul" else "xw")]
            shape = cast(Node, bound["q_a" if name == "accumulate_matmul" else "q_w"]).meta["val"].shape
            count = math.prod(shape[1:]) if name == "accumulate_conv2d" else shape[-1]
            if accumulate_bits(sides[0], sides[1], count, peak) <= 32:
                bound["out_format"] = ACCUMULATOR_FORMAT.name
            biases.update({user.name: cast(str, bound["out_format"]) for user in adds})
        
        # Other arguments are attributes: lists, numbers, strings, and torch types by name (devices are left out)
        for argument in schema.arguments:
            value = bound[argument.name]
            if "Tensor" not in str(argument.type):
                value = known[value.name] if isinstance(value, Node) else value
                value = [*value] if isinstance(value, (list, tuple)) else str(value) if isinstance(value, (torch.dtype, torch.memory_format, torch.layout)) else value
                if value is not None and not isinstance(value, torch.device):
                    attributes[argument.name] = value
                continue
            
            # Tensor arguments are inputs (None for None). A list of tensors also says its length
            if isinstance(value, (list, tuple)):
                attributes[f"{argument.name}_count"] = len(value)
            for position, item in enumerate(value if isinstance(value, (list, tuple)) else [value]):
                source = item.name if isinstance(item, Node) else f"{node.name}_{argument.name}_{position}"
                if item is None or (isinstance(item, Node) and item.name not in known) or source in labels:
                    names.append(None if item is None else labels.get(source, alias.get(source, rename.get(source, source))))
                    roles.append(EXPORT_ROLES.get(name if domain == "quaver" else f"{domain}.{name}", {}).get(argument.name, ""))
                    continue
                
                # A new data item, named by its first use: the path of the layer and the role of the argument (a second use of one role in a layer adds _2)
                role           = EXPORT_ROLES.get(name if domain == "quaver" else f"{domain}.{name}", {}).get(argument.name, f"{name}_{argument.name}")
                label          = f"{chain[-1][1]}.{role}" if chain else role
                label          = label if label not in constants else next(f"{label}_{count}" for count in range(2, len(constants) + 3) if f"{label}_{count}" not in constants)
                labels[source] = label
                tensor         = torch.as_tensor(known[item.name] if isinstance(item, Node) else item).detach().cpu()
                format_name    = (cast(str, bound[spec[argument.name]]) if spec[argument.name] in bound else spec[argument.name]) if argument.name in spec else ""
                if not format_name:
                    format_name = biases.get(node.name) or (EXPORT_TYPES[tensor.dtype] if tensor.dtype.is_floating_point else "int32" if tensor.numel() == 0 or (-2**31 <= int(tensor.min()) and int(tensor.max()) < 2**31) else "int64")
                
                # The data item (its role keeps the _2 of a second use)
                constants[label] = {"role": label.removeprefix(f"{chain[-1][1]}.") if chain else label, "tensor": tensor, "format": format_name, "path": chain[-1][1] if chain else ""}
                names.append(label)
                roles.append(role)
        
        # Outputs: one value, or the values that getitem takes out of a tuple (None for the parts that nothing uses)
        if isinstance(node.meta["val"], (list, tuple)):
            parts   = {user.args[1]: user for user in node.users if user.target is operator.getitem}
            outputs = [parts[index].name if index in parts else None for index in range(len(node.meta["val"]))]
            shapes  = list(node.meta["val"])
        else:
            outputs = [node.name]
            shapes  = [node.meta["val"]]
        
        # The format of each output: from the op, or kept from the first input that is a value of the same type (with its scale and zero point when the shape stays or the scale is one number)
        first = next((item.meta.get("val") for item in node.all_input_nodes if item.name not in known)                                                    , None)
        kept  = next((metadata.get(alias.get(item.name, rename.get(item.name, item.name)), {}) for item in node.all_input_nodes if item.name not in known), {}  )
        for index, (output, value) in enumerate(zip(outputs, shapes)):
            if not output or not isinstance(value, torch.Tensor):
                continue
            if index in spec:
                metadata[output] = {"format": cast(str, bound[spec[index]]) if spec[index] in bound else spec[index]}
                continue
            if not (isinstance(first, torch.Tensor) and first.dtype == value.dtype and kept):
                metadata[output] = {"format": EXPORT_TYPES[value.dtype]}
                continue
            metadata[output] = {key: item for key, item in kept.items() if key == "format" or tuple(first.shape) == tuple(value.shape) or not isinstance(item, list)}
        
        # The op step (with the overload of a torch op that is not the default one)
        step = {"op": f"{domain}.{name}", "inputs": names, "outputs": outputs, "attributes": attributes}
        if getattr(node.target, "_overloadname", "default") != "default":
            step["overload"] = getattr(node.target, "_overloadname")
        records.append({"step": step, "inputs": names, "roles": roles, "outputs": outputs, "chain": chain, "groups": []})
        named_outputs.update({cast(str, outputs[index]): role for index, role in EXPORT_OUTPUTS.get(name if domain == "quaver" else "", {}).items() if outputs[index]})
    
    # The outputs of the model (the codes of a pair carry the note of its parameters)
    outputs_needed = [alias.get(node.name, node.name) for node in results]
    
    # Group the records into layer calls: a run of records in one call of a layer is one group, and groups nest like the modules
    root  : dict           = {"id": 0, "items": [], "overload": "", "layout": EXPORT_LAYOUTS[(layout or {}).get("", "channels_last")]}
    groups: list[dict]     = [root]
    calls : dict[str, int] = {}
    for record in records:
        parent = root
        
        # Walk down the chain, and start a new group where a new call of a layer begins. A layer takes its layout from its path, its class, or its block
        for key, place, kind in record["chain"]:
            last   = parent["items"][-1] if parent["items"] else None
            if last is None or last.get("key") != key:
                calls[place] = calls.get(place, 0) + 1
                chosen       = (layout or {}).get(place, (layout or {}).get(kind))
                last         = {"id": len(groups), "key": key, "kind": kind, "overload": place if calls[place] == 1 else f"{place}@{calls[place]}", "module": wrapped.get_submodule(f"model.{place}"), "items": [],
                                "layout": EXPORT_LAYOUTS[chosen] if chosen else parent["layout"]}
                groups.append(last)
                parent["items"].append(last)
            parent = last
            record["groups"].append(parent["id"])
        parent["items"].append(record)
    
    # The inputs of each group (read inside, made outside) and its outputs (made inside, and read outside or by the model)
    for group in groups[1:]:
        inside           = [record for record in records if group["id"] in record["groups"]]
        made             = {name for record in inside for name in record["outputs"] if name}
        read             = {name for record in records if group["id"] not in record["groups"] for name in record["inputs"]} | set(outputs_needed)
        group["inputs"]  = list(dict.fromkeys(name for record in inside for name in record["inputs"] if name and name not in made and name not in constants))
        group["outputs"] = list(dict.fromkeys(name for record in inside for name in record["outputs"] if name in read)                                      )
    
    # Readable names for the values that layers give: "path.output" (then "path.output_role" or "path.output_2"), outer layers first
    edges: dict[str, str] = {}
    for group in groups[1:]:
        for index, name in enumerate(group["outputs"]):
            edges.setdefault(name, f"{group['overload']}.output" + ("" if index == 0 else f"_{named_outputs[name]}" if name in named_outputs else f"_{index + 1}"))
    for record in records:
        record["step"]["inputs"]  = [edges.get(name, name) if name else None for name in record["step"]["inputs"]]
        record["step"]["outputs"] = [edges.get(name, name) if name else None for name in record["step"]["outputs"]]
        record["inputs"]          = record["step"]["inputs"]
    for group in groups[1:]:
        group["inputs"], group["outputs"] = [edges.get(name, name) for name in group["inputs"]], [edges.get(name, name) for name in group["outputs"]]
    
    # The outputs of the model under these names, and the record that makes each value
    outputs_needed = [edges.get(name, name) for name in outputs_needed]
    makers         = {name: record for record in records for name in record["outputs"] if name}
    
    # The format, the shape, and the parameters of every needed value, under its final name
    values: dict[str, dict[str, Any]] = {}
    for node in program.graph.nodes:
        value = node.meta.get("val")
        name  = str(rename.get(node.name, node.name))
        if node.name in needed and isinstance(value, torch.Tensor):
            if value.dtype not in EXPORT_TYPES:
                raise ValueError(f"The value {node.name} has the type {value.dtype}, which the export does not handle.")
            values[edges.get(name, name)] = {"name": edges.get(name, name), **metadata.get(name, {}), "shape": list(value.shape)}
    
    # The settings of each layer call: the settings of its module, its shape, its layout, the formats of its first input and output, and the range of the output codes
    for group in groups[1:]:
        module   = group["module"]
        settings = {key: getattr(module, key) for key in EXPORT_SETTINGS if isinstance(getattr(module, key, None), (bool, int, str, tuple, list))}
        weight   = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.ndim >= 3:
            settings.update(out_channels=weight.shape[0], in_channels=weight.shape[1], kernel_size=list(weight.shape[2:]))
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            settings.update(out_features=weight.shape[0], in_features=weight.shape[1])
        if isinstance(getattr(getattr(module, "blocks", None), "block_size", None), int):
            settings["block_size"] = getattr(module, "blocks").block_size
        settings.update(layout=group["layout"])
        
        # The formats of the first input, of the first output, and of the weight, and the range of the output codes
        sources = {"input_format": values.get(group["inputs"][0], {}), "output_format": values.get(group["outputs"][0], {}), "weight_format": constants.get(f"{group['overload']}.weight", {})}
        settings.update({key: source["format"] for key, source in sources.items() if source.get("format")})
        if settings.get("output_format"):
            named = build_format(str(settings["output_format"]))
            settings.update(output_min=round(named.min_value / named.step_min), output_max=round(named.max_value / named.step_min))
        
        # Lists for tuples, and whether the layer works on images (it has a kernel)
        group["settings"] = {key: [*value] if isinstance(value, (tuple, list)) else value for key, value in settings.items()}
        group["spatial"]  = "kernel_size" in group["settings"]
    for group in reversed(groups[1:]):
        group["spatial"] = group["spatial"] or any(item.get("spatial") for item in group["items"])
    
    # The layout of the images at the inputs and the outputs of layers (rank 4 or more, and rank 3 at layers with a kernel): channels_last stores (batch, spatial..., channel)
    for group in groups[1:]:
        for name in [*group["inputs"], *group["outputs"]]:
            rank    = len(values[name]["shape"]) if name in values else 0
            spatial = EXPORT_SPATIAL.get(rank - 2, [f"s{index}" for index in range(rank - 2)]) if rank >= 4 or (rank == 3 and group["spatial"]) else None
            if spatial is not None and "layout" not in values[name]:
                producer               = makers[name]["groups"][-1] if name in makers and makers[name]["groups"] else group["id"]
                chosen                 = groups[producer]["layout"]
                values[name]["layout"] = chosen
                values[name]["axes"]   = ["batch", *spatial, "channel"] if chosen == "channels_last" else ["batch", "channel", *spatial]
    
    # The data items: each one in its stored order (layout, blocks once, columns of a flattened input reordered), packed back to back with the alignment
    blob    = bytearray()
    entries = {group["overload"]: [] for group in groups}
    for label, constant in constants.items():
        group  = next((group for group in groups if group["overload"] == constant["path"]), root)
        tensor = constant["tensor"]
        item   = {"role": constant["role"], "name": label, "format": constant["format"], "shape": list(tensor.shape)}
        
        # Weights with spatial axes: channels_last keeps (out, spatial..., in). Block codes are kept once for each block of the input axis
        if constant["role"] in ("weight", "weight_block_codes") and tensor.ndim >= 3:
            size    = group["settings"].get("block_size") if constant["role"] == "weight_block_codes" else None
            if size and tensor.shape[1] % size == 0 and tensor.shape[1] > 1:
                tensor         = tensor.unflatten(1, (-1, size))[:, :, 0]
                item["blocks"] = {"axis": 1, "size": size}
            spatial = EXPORT_SPATIAL.get(tensor.ndim - 2, [f"s{index}" for index in range(tensor.ndim - 2)])
            inner   = "in_block" if "blocks" in item else "in"
            if group["layout"] == "channels_last":
                tensor         = tensor.permute(0, *range(2, tensor.ndim), 1)
                item["layout"] = "channels_last"
                item["axes"]   = ["out", *spatial, inner]
            else:
                item["layout"] = "channels_first"
                item["axes"]   = ["out", inner, *spatial]
        
        # Linear weights: (out, in). A flatten of channels_last images before the layer reorders the columns to (spatial..., channel), unless every spatial axis has size 1
        elif constant["role"] in ("weight", "weight_block_codes") and tensor.ndim == 2:
            maker  = makers.get(group["inputs"][0]) if group["inputs"] else None
            source = values.get(cast(str, maker["inputs"][0])) if maker and maker["step"]["op"] == "aten.flatten" else None
            if source and len(source["shape"]) >= 3 and source.get("layout") == "channels_last" and any(size > 1 for size in source["shape"][2:]):
                if constant["role"] == "weight_block_codes" or "block_size" in group["settings"]:
                    raise ValueError(f"{group['overload']}: a block-scaled Linear after a flatten of channels_last images would mix its blocks. Give the layer before it the layout channels_first.")
                channels, *spatial_sizes = source["shape"][1:]
                tensor                   = tensor.unflatten(1, (channels, *spatial_sizes)).permute(0, *range(2, 2 + len(spatial_sizes)), 1).flatten(1)
                spatial                  = EXPORT_SPATIAL.get(len(spatial_sizes), [f"s{index}" for index in range(len(spatial_sizes))])
                item["reorder"]          = {"from": ["channel", *spatial], "to": [*spatial, "channel"], "shape": [channels, *spatial_sizes]}
            
            # The weight of a Linear is (out, in) in both layouts
            item["layout"] = "row_major"
            item["axes"]   = ["out", "in"]
        
        # Other data (bias, zero points, M0 and n, clamps, scales): the axes of size 1 are left out
        else:
            kept           = [index for index, size in enumerate(tensor.shape) if size != 1]
            tensor         = tensor.reshape([tensor.shape[index] for index in kept])
            item["layout"] = "flat"
            item["axes"]   = ["out" if constant["role"].startswith("weight") else "channel"] if len(kept) == 1 else [f"axis{index}" for index in kept]
        
        # Pack it at the next aligned offset
        packed, encoding = pack_codes(tensor.contiguous(), constant["format"])
        blob.extend(bytes(-len(blob) % alignment))
        item.update(stored_shape=list(tensor.shape), **encoding, offset=len(blob), size=len(packed))
        blob.extend(packed)
        entries[group["overload"]].append(item)
        
        # The same numbers in decimal, in the stored order: INT codes as integers, FP codes as the real values of the format, floats as they are
        if json_values:
            named          = None if constant["format"] in ("float32", "float64") else build_format(constant["format"])
            item["values"] = tensor.double().tolist() if named is None else (tensor.double() * named.step_min).tolist() if named.kind == "fp" else tensor.long().tolist()
    
    # The JSON of each layer: its class, inputs and outputs with their formats and shapes, settings, data items, and its children ("layers", with the ops of the block between them) or its ops
    for group in reversed(groups):
        steps = [item["step"] if "step" in item else item["json"] for item in group["items"]]
        if group is root:
            continue
        
        # A block holds its children and the ops between them in "layers", and a layer without children holds its ops in "ops"
        group["json"] = {"name": group["overload"], "class": group["kind"], "inputs": [values[name] for name in group["inputs"]], "outputs": [values[name] for name in group["outputs"]], "settings": group["settings"],
                         "data": entries[group["overload"]], ("layers" if any("json" in item for item in group["items"]) else "ops"): steps}
    
    # The model: the file format, the inputs and the outputs, the data items outside the layers, the layers, and every value
    exported = {
        "quaver"     : 1,
        "alignment"  : alignment,
        "byte_order" : "little",
        "json_values": json_values,
        "size"       : len(blob),
        "codes"      : "INT codes are two's complement, FP codes bit patterns (sign, exponent, mantissa). 2-bit and 4-bit codes fill each byte from the low bits up. An offset is the address in the file. values: the same numbers in decimal.",
        "inputs"     : [values[rename.get(str(name), str(name))] for name in signature.user_inputs if rename.get(str(name), str(name)) in values],
        "outputs"    : [values[str(name)] for name in outputs_needed],
        "data"       : entries[""],
        "layers"     : [item["step"] if "step" in item else item["json"] for item in root["items"]],
        "values"     : values,
    }
    
    # The JSON text: a list of only numbers or strings stays on one line (one row of "values" is one line), and the rest is indented. A copy holds marks in place of those lists
    marked = json.loads(json.dumps(exported))
    rows   = []
    stack  = [marked]
    while stack:
        node = stack.pop()
        for key, child in list(node.items() if isinstance(node, dict) else enumerate(node)):
            if isinstance(child, list) and not any(isinstance(value, (dict, list)) for value in child):
                node[key] = f"\0{len(rows)}"
                rows.append(child)
            elif isinstance(child, (dict, list)):
                stack.append(child)
    text   = re.sub(r'"\\u0000(\d+)"', lambda found: json.dumps(rows[int(found.group(1))]), json.dumps(marked, indent=1))
    
    # The two files
    target = Path(folder)
    target.mkdir(parents=True, exist_ok=True)
    (target / "model.json").write_bytes(text.encode("utf-8"))
    (target / "model.bin").write_bytes(bytes(blob))
    return exported
