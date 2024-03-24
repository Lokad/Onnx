import os
import onnx

from typing import Any, Optional, Sequence, Tuple, Dict
import numpy as np
from onnx.backend.base import Backend, BackendRep, namedtupledict

from . import util
from . import tensors
from . import lokadonnx

class LokadOnnxRep(BackendRep):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph

    def run(self, inputs, **kwargs):
        r = False
        if isinstance(inputs, dict):
            r = self.graph.Execute(inputs)
        elif isinstance(inputs, list):
            _inputs = list(map(tensors.make_tensor_from_ndarray, inputs))
            r = self.graph.Execute(tensors.make_tensor_array(_inputs))
        else:
            raise RuntimeError(f'The input type {type(inputs)} is not supported by the backend.')
        if not r:
            raise RuntimeError('The graph did not execute successfully.')
        
        outputs = self.convert_graph_outputs(self.graph.Outputs, 'Outputs')
        self.graph.Reset()
        return outputs
    
    def run_node(self, inputs, node_name:str, **kwargs):
        r = False
        if isinstance(inputs, dict):
            r = self.graph.ExecuteNode(inputs, node_name)
        elif isinstance(inputs, list):
            _inputs = list(map(tensors.make_tensor_from_ndarray, inputs))
            r = self.graph.ExecuteNode(tensors.make_tensor_array(_inputs), node_name)
        else:
            raise RuntimeError(f'The input type {type(inputs)} is not supported by the backend.')
        if not r:
            raise RuntimeError('The graph did not execute successfully.')
        
        outputs = self.convert_graph_outputs(self.graph.Outputs, 'Outputs')
        self.graph.Reset()
        return outputs

    def convert_graph_outputs(self, dict, name:str):
        keys = []
        values = []
        for kv in dict:
            keys.append(kv.Key)
            values.append(tensors.make_ndarray_from_tensor(kv.Value))
        return namedtupledict(name, keys)(*values)

class LokadOnnxBackend(Backend):
    @classmethod
    def is_compatible(cls, model: onnx.ModelProto, device: str = "CPU", **kwargs) -> bool:
        if device == 'CPU':
            return True
        else:
            return False
    
    @classmethod
    def supports_device(cls, device: str) -> bool:
        return device == 'CPU'
    
    @classmethod
    def prepare(cls, model:onnx.ModelProto, device:str='CPU', **kwargs) -> LokadOnnxRep:
        super(LokadOnnxBackend, cls).prepare(model, device, **kwargs)
        name = model.graph.name + "_" + util.generate_random_filename()
        onnx.save(model, name)
        graph = lokadonnx.load_graph(name)
        os.remove(name)
        return LokadOnnxRep(graph)
    
    @classmethod
    def run_node(cls, node: onnx.NodeProto, inputs: Any, device: str = "CPU", outputs_info: Optional[Sequence[Tuple[np.dtype, Tuple[int, ...]]]] = None,**kwargs: Dict[str, Any],) -> Optional[Tuple[Any, ...]]:
        super(LokadOnnxBackend, cls).run_node(node, inputs, device, outputs_info)
        graph_inputs = []
        graph_outputs = []
        if isinstance(inputs, list):
            for n, i in enumerate(node.input):
                graph_inputs.append(onnx.helper.make_tensor_value_info(i, onnx.helper.np_dtype_to_tensor_dtype(inputs[n].dtype), inputs[n].shape))
        elif isinstance(inputs, dict):
            for name, val in inputs.iteritems():
                graph_inputs.append(onnx.helper.make_tensor_value_info(name, onnx.helper.np_dtype_to_tensor_dtype(val.dtype), val.shape))
        else:
            raise TypeError(f'The type of inputs: {type(inputs)} is not supported.')
        
        graph_outputs = map(lambda i: onnx.helper.make_tensor_value_info(i, onnx.TensorProto.INT32, [0]), node.output)
        graph = onnx.helper.make_graph([node], node.name + "_graph", graph_inputs, graph_outputs)
        model = onnx.helper.make_model(graph)
        rep = prepare(model)
        return rep.run_node(inputs, node.name)

def prepare_file(file_path:str) -> LokadOnnxRep:
     graph = lokadonnx.load_graph(file_path)
     return LokadOnnxRep(graph)

prepare = LokadOnnxBackend.prepare

run_node = LokadOnnxBackend.run_node

run_model = LokadOnnxBackend.run_model

supports_device = LokadOnnxBackend.supports_device