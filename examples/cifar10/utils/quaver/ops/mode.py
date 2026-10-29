# @file mode.py
# @brief Pick the float, training, or integer version of every op
# @author Kohei Yamaguchi, Claude

from .activation import activation_fake, activation_float, activation_integer
from .add        import add_fake, add_float, add_integer
from .block      import block_quantize_fake, block_quantize_float, block_quantize_integer
from .matmul     import matmul_fake, matmul_float, matmul_integer
from .normalize  import normalize_l1_fake, normalize_l1_float, normalize_l1_integer, normalize_l1_scores_fake, normalize_l1_scores_float, normalize_l1_scores_integer
from .pool       import pool_avg2d_fake, pool_avg2d_float, pool_avg2d_integer
from .requantize import requantize_fake, requantize_float, requantize_integer
from .types      import OpSet

FLOAT_OPS   = OpSet(requantize=requantize_float  , add=add_float  , pool_avg2d=pool_avg2d_float  , matmul=matmul_float  , normalize_l1=normalize_l1_float  , normalize_l1_scores=normalize_l1_scores_float  , activation=activation_float  ,
                    block_quantize=block_quantize_float  )
FAKE_OPS    = OpSet(requantize=requantize_fake   , add=add_fake   , pool_avg2d=pool_avg2d_fake   , matmul=matmul_fake   , normalize_l1=normalize_l1_fake   , normalize_l1_scores=normalize_l1_scores_fake   , activation=activation_fake   ,
                    block_quantize=block_quantize_fake   )
INTEGER_OPS = OpSet(requantize=requantize_integer, add=add_integer, pool_avg2d=pool_avg2d_integer, matmul=matmul_integer, normalize_l1=normalize_l1_integer, normalize_l1_scores=normalize_l1_scores_integer, activation=activation_integer,
                    block_quantize=block_quantize_integer)



def mode_ops(integer_mode: bool, float_mode: bool = False) -> OpSet:
    '''
    #* Return the ops of the integer mode, the training mode, or the float mode.
    #? A layer writes its forward pass once with these ops. The training and integer modes run the same integer math. The float mode (calibration) runs plain float ops.
    #! integer_mode wins over float_mode.
    '''
    
    if integer_mode:
        return INTEGER_OPS
    return FLOAT_OPS if float_mode else FAKE_OPS
