# @file values.py
# @brief The values to compute with, for each data type (FR-039, 02a 4章)

import torch



VALUES_FLOAT32 = frozenset({torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz, torch.float8_e8m0fnu})
VALUES_INTEGER = frozenset({torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.uint16, torch.uint32, torch.uint64})
VALUES_COMPLEX = frozenset({torch.complex32, torch.complex64, torch.complex128})
VALUES_QUANT   = frozenset({torch.qint8, torch.quint8, torch.qint32, torch.quint4x2, torch.quint2x4})
VALUES_DTYPES  = frozenset({torch.float32, torch.float64, torch.bool}) | VALUES_FLOAT32 | VALUES_INTEGER | VALUES_COMPLEX | VALUES_QUANT



def values_problem(tensor: torch.Tensor) -> str | None:
    '''
    #* Return why a tensor cannot be recorded, or None when it can (MSG-012).
    '''
    
    if tensor.layout != torch.strided:
        return "sparse テンソル"
    if tensor.is_nested:
        return "nested テンソル"
    if tensor.is_meta:
        return "meta デバイスのテンソル"
    if tensor.dtype not in VALUES_DTYPES:
        return "データ型 " + str(tensor.dtype).removeprefix("torch.")
    return None


def values_prepare(tensor: torch.Tensor) -> torch.Tensor:
    '''
    #* Return the values to compute the statistics with: float32 or float64, on the same device and with the same shape.
    #! Pass only tensors for which values_problem() returns None. The result may share memory with the tensor; nobody writes to it.
    '''
    
    # Quantized tensors give their real values
    tensor = tensor.detach()
    if tensor.is_quantized:
        return tensor.dequantize()
    
    # Floats stay or become float32, integers become float64, and bool becomes 1 and 0
    if tensor.dtype in (torch.float32, torch.float64):
        return tensor
    if tensor.dtype in VALUES_FLOAT32 or tensor.dtype == torch.bool:
        return tensor.to(torch.float32)
    if tensor.dtype in VALUES_INTEGER:
        return tensor.to(torch.float64)
    
    # Complex numbers give their absolute values
    if tensor.dtype == torch.complex32:
        return tensor.to(torch.complex64).abs()
    return tensor.abs()
