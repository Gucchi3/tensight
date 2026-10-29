# @file inspect.py
# @brief Summarize one tensor without a run (02a API-09, ADR-014)
# @author Kohei Yamaguchi, Claude

import torch

from .adapters import registry_adapters
from .core     import message_warn
from .stats    import pattern_default, pattern_parse, pattern_plan, quant_summary, summary_move, summary_parts, summary_report, summary_tensor, values_prepare, values_problem
from .types    import MapPattern, QuantPair, QuantValue, TensorReport



def inspect_tensor(value: object, *, before: torch.Tensor | None = None, map: str | None = None, bins: int = 64, sample_size: int = 16) -> TensorReport:
    '''
    #* Return the summary of one tensor with the same computation as the records, without writing files (02a API-09, IF-09).
    #! It does not touch the run: the step, the collector and the files stay as they are.
    '''
    
    # Check the numbers and the map pattern (02a API-09)
    for key, number, low in (("bins", bins, 1), ("sample_size", sample_size, 0)):
        if not isinstance(number, int) or isinstance(number, bool):
            raise TypeError("引数 " + key + " は int にしてください。渡された値: " + repr(number))
        if not low <= number <= 4096:
            raise ValueError("引数 " + key + " は " + str(low) + " 以上 4096 以下 にしてください。渡された値: " + repr(number))
    if map is not None and not isinstance(map, str):
        raise TypeError("引数 map は str か None にしてください。渡された値: " + repr(map))
    spec = pattern_parse(map) if map is not None else None
    
    # Read the quantization with the readers, in their order
    quant: QuantValue | None = None
    for adapter in registry_adapters():
        try:
            quant = adapter.read_value(value)
        except Exception as error:
            message_warn("MSG-003", adapter=adapter.name, error=error)
            quant = None
        if quant is not None:
            break
    
    # Find the tensor, and check that it can be summarized
    if isinstance(value, torch.Tensor):
        tensor = value
    elif quant is not None:
        tensor = quant.source if quant.source is not None else quant.values
    else:
        raise TypeError("inspect_tensor に渡せるのは、テンソルか、読み取り部品が扱う値です。入れ子の値は、中のテンソルごとに呼んでください")
    problem = values_problem(tensor)
    if problem is not None:
        raise TypeError(problem + "は集計できません")
    
    # Check the values before the quantizer
    if before is not None:
        if quant is None:
            raise ValueError("before は、量子化されたテンソルを渡したときだけ使えます")
        if not isinstance(before, torch.Tensor):
            raise TypeError("引数 before は torch.Tensor にしてください。渡された値: " + repr(before))
        if before.shape != quant.values.shape:
            raise ValueError("before の形 " + str(list(before.shape)) + " が、value の形 " + str(list(quant.values.shape)) + " と違います")
    
    # Fit the map pattern; "auto" uses the fixed pattern without a warning, because there is no layer
    pattern = pattern_default(tensor.ndim) if spec == "auto" else spec if isinstance(spec, MapPattern) else None
    plan    = pattern_plan(pattern, tuple(tensor.shape)) if pattern is not None else None
    if pattern is not None and plan is None:
        named = len([name for name in pattern.left if name != "..."])
        raise ValueError("並べ方の式「" + pattern.text + "」は " + str(named) + " 次元" + ("以上" if "..." in pattern.left else "") + "のテンソル用です。渡されたテンソルは " + str(tensor.ndim) + " 次元です")
    
    # Summarize with the same functions as the records, and move the result to the CPU at once (ADR-014)
    with torch.no_grad():
        values  = values_prepare(quant.values if quant is not None else tensor)
        pending = summary_tensor(values, tensor, bins, sample_size, None, (), plan, pattern, False, False)
        found   = QuantPair(before, quant.values, quant.info) if before is not None and quant is not None else quant
        summary = quant_summary(found, values) if found is not None else None
        return summary_report(pending, summary, summary_move(summary_parts(pending, summary)))
