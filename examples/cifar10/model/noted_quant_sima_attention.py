# @file noted_quant_sima_attention.py
# @brief The SimA attention of quaver that also notes its quantized attention map for tensight
# @author Kohei Yamaguchi, Claude

import tensight
import torch

from utils.quaver     import QuantParams, QuantSimaAttention
from utils.quaver.ops import mode_ops



class NotedQuantSimaAttention(QuantSimaAttention):
    '''
    #* The SimA attention of quaver that notes its quantized attention map as "block.attention.map", the point name of the FP32 model (SimaAttention).
    #? QuantSimaLayer uses this, so the QAT run gets the same point as the FP32 run, and the viewer can compare the two maps.
    #! forward() is the forward of QuantSimaAttention (utils/quaver/samples/quant_sima_attention.py) with one note added, and the values do not change. Keep it the same as quaver when quaver changes.
    '''
    
    def forward(self, x: torch.Tensor, x_params: QuantParams) -> tuple[torch.Tensor, QuantParams]:
        '''
        #* Run SimA attention as quaver does, and note the attention map: fake values in the training mode, codes in the integer mode.
        #! In the float mode (calibration) the map is not quantized, so it is noted without its parameters. Outside a run (for example the tests after training) nothing is noted, so tensight gives no warning.
        '''
        
        # Q and K: 1x1 conv to the INT32 accumulator, cut into heads, and NEQ (one scale for each sample and head, or for each token)
        ops                     = mode_ops(self.integer_mode, self.float_mode)
        batch, _, height, width = x.shape
        query, query_params     = self._sima_tokens(self.query, x, x_params, ops)
        key, key_params         = self._sima_tokens(self.key  , x, x_params, ops)
        
        # Attention map: exact Q K^T, requantized with the integer maxima of NEQ, and 1 / sqrt(key_dim) in the static part of the multiplier
        factor               = self.key_dim**-0.5
        key_t_params         = QuantParams(key_params.scale.transpose(-1, -2), key_params.zero_point.transpose(-1, -2), key_params.number_format, peak=None if key_params.peak is None else key_params.peak.transpose(-1, -2))
        scores, score_params = ops.matmul(query, query_params, key.transpose(-1, -2), key_t_params)
        attention_params     = self.attention_quantizer.params(scores) if self.integer_mode else self.attention_quantizer.measure(scores * factor)
        attention            = ops.normalize_l1_scores(scores, score_params, query_params, key_t_params, attention_params, factor, self.style)
        
        # tensight records the map (batch, heads, query, key) as a query x key map, with the quantization of the map, in the recorded steps of a run
        if tensight.run_should_record():
            tensight.note_tensor("block.attention.map", attention if self.float_mode else (attention, attention_params), map="attention_qk")
        
        # Values: 1x1 conv to the activation format, cut into heads
        value, value_params = self.value(x, x_params)
        value               = value.reshape(batch, self.heads, self.value_dim, -1).transpose(-1, -2)
        
        # Output: exact attention x V, the activation on the accumulator (none by default), then requantize to the projection input (static M0 and n)
        output, output_params = ops.matmul(attention, attention_params, value, value_params)
        output                = ops.activation(output, output_params, self.activation)
        project_params        = self.project_quantizer.params(output) if self.integer_mode else self.project_quantizer.measure(output)
        output                = ops.requantize(output, output_params, project_params, self.style)
        output                = output.transpose(-1, -2).reshape(batch, self.heads * self.value_dim, height, width)
        return self.project(output, project_params)
