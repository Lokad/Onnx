import os
import onnx

from . import util
from . import tensors
from . import lokadonnx

from onnx.backend.base import Backend, BackendRep

class LokadOnnxRep(BackendRep):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph

    def run(self, inputs, **kwargs):
        r = False
        if isinstance(inputs, dict):
            r = self.graph.Execute(inputs)
        elif isinstance(inputs, list):
            r = self.graph.Execute(tensors.make_tensor_array(*inputs))
        else:
            raise RuntimeError(f'The input type {type(inputs)} is not supported by the backend.')
        if not r:
            raise RuntimeError('The graph did not execute successfully.')
        
        outputs = util.convert_dictionary_to_namedtupledict(self.graph.Output, 'Outputs')
        self.graph.Reset()
        return outputs

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
        name = util.generate_random_filename()
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