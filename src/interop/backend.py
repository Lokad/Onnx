import os
import onnx

from typing import Any
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
    def prepare(cls, model:onnx.ModelProto, device:str='CPU', **kwargs) -> LokadOnnxRep:
        super(LokadOnnxBackend, cls).prepare(model, device, **kwargs)
        name = model.graph.name + "_" + util.generate_random_filename()
        onnx.save(model, name)
        graph = lokadonnx.load_graph(name)
        os.remove(name)
        return LokadOnnxRep(graph)

def prepare_file(file_path:str) -> LokadOnnxRep:
     graph = lokadonnx.load_graph(file_path)
     return LokadOnnxRep(graph)

prepare = LokadOnnxBackend.prepare


#run_node = LokadOnnxBackend.run_node

#run_model = LokadOnnxBackend.run_model

#supports_device = LokadOnnxBackend.supports_device