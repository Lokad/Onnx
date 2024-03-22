import sys,os
import random
import string
import onnx
import tempfile

from . import lokadonnx

from onnx.backend.base import Backend, BackendRep

def generate_random_filename(length: int = 24, extension: str = "") -> str:
    """Generates a random filename"""
    characters = string.ascii_letters + string.digits
    random_string = "".join(random.choice(characters) for _ in range(length))
    if extension:
        if "." in extension:
            pieces = extension.split(".")
            last_extension = pieces[-1]
            extension = last_extension
        return f"{random_string}.{extension}"
    return random_string
        

class LokadOnnxRep(BackendRep):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph

class LokadOnnxBackend(Backend):
    @classmethod
    def is_compatible(self, model: onnx.ModelProto, device: str = "CPU", **kwargs) -> bool:
        if device == 'CPU':
            return True
        else:
            return False
    
    @classmethod
    def prepare(self, model:onnx.ModelProto, device:str='CPU', **kwargs) -> LokadOnnxRep:
        super(LokadOnnxBackend, self).prepare(model, device, **kwargs)
        name = generate_random_filename()
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