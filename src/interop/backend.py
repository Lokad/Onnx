import os
import onnx

from typing import Any, Optional, Sequence, Tuple, Dict
import numpy as np
from onnx.backend.base import Backend, BackendRep, namedtupledict

from . import util
from . import tensors

import clr
file_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
clr.AddReference(os.path.join(file_dir, "..", "Lokad.Onnx.Interop", "bin", "Release", "net6.0", "publish", "Lokad.Onnx.Interop.dll"))
clr.AddReference(os.path.join(file_dir, "..", "Lokad.Onnx.Interop", "bin", "Release", "net6.0", "publish", "Lokad.Onnx.Tensors.dll"))
clr.AddReference(os.path.join(file_dir, "..", "Lokad.Onnx.Interop", "bin", "Release", "net6.0", "publish", "Lokad.Onnx.Backend.dll"))
clr.AddReference(os.path.join(file_dir, "..", "Lokad.Onnx.Interop", "bin", "Release", "net6.0", "publish", "Lokad.Onnx.Data.dll"))

from System import Array, String
from System.Collections.Generic import Dictionary
from Lokad.Onnx import ITensor, ComputationalGraph
from Lokad.Onnx.Interop import Tensors,Graph
        
class LokadOnnxRep(BackendRep):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph

    def run(self, inputs, **kwargs):
        file_args = kwargs['file_args'] if 'file_args' in kwargs else []
        save_file_arg = kwargs['save_file_arg'] if 'save_file_arg' in kwargs else False
        if not isinstance(file_args, list):
            raise TypeError(f'The file_args argument must be type list, not {type(file_args)}')
        if not isinstance(save_file_arg, bool):
            raise TypeError(f'The save_file_arg argument must be type bool, not {type(save_file_arg)}')
        r = False
        if isinstance(inputs, dict):
            _inputs = {}
            for key, value in inputs.items():
                if not isinstance(key, str) or not isinstance(value, np.ndarray):
                    raise TypeError(f'An input dictionary item must have type str:ndarray or str:str, not {type(key)}:{type(value)}.')
                if key in file_args:
                    _inputs[key] = Graph.GetInputTensorFromFileArg(value, save_file_arg)
                else:
                    _inputs[key] = tensors.make_tensor_from_ndarray(value)
            r = self.graph.Execute(tensors.make_tensor_dictionary(_inputs))
        elif isinstance(inputs, list):
            _inputs=[]
            for index,value in enumerate(inputs):
                if index in file_args:
                    _inputs.append(Graph.GetInputTensorFromFileArg(value, save_file_arg))
                else:
                    _inputs.append(tensors.make_tensor_from_ndarray(value))
            r = self.graph.Execute(tensors.make_tensor_array(_inputs))
        else:
            raise RuntimeError(f'The input type {type(inputs)} is not supported by the backend.')
        if not r:
            raise RuntimeError('The graph did not execute successfully.')
        outputs = self.convert_graph_outputs(self.graph.Outputs, 'Outputs')
        self.graph.Reset()
        return outputs
    
    def run_graph_node(self, inputs, node_name:str):
        r = False
        if isinstance(inputs, Array[ITensor]):
            r = self.graph.ExecuteNode(inputs, node_name)
        elif isinstance(inputs, Dictionary[string, ITensor]):
            r = self.graph.ExecuteNode(inputs, node_name)
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
        graph = load_graph(name)
        os.remove(name)
        return LokadOnnxRep(graph)
    
    @classmethod
    def run_node(cls, node: onnx.NodeProto, inputs: Any, device: str = "CPU", outputs_info: Optional[Sequence[Tuple[np.dtype, Tuple[int, ...]]]] = None,**kwargs: Dict[str, Any],) -> Optional[Tuple[Any, ...]]:
        super(LokadOnnxBackend, cls).run_node(node, inputs, device, outputs_info)
        graph_inputs = []
        graph_outputs = []
        node_inputs_list = []
        node_inputs_dict = {}
        file_args = kwargs['file_args'] if 'file_args' in kwargs else []
        save_file_arg = kwargs['save_file_arg'] if 'save_file_arg' in kwargs else False
        if not isinstance(file_args, list):
            raise TypeError(f'The file_args argument must be type list, not {type(file_args)}')
        if not isinstance(save_file_arg, bool):
            raise TypeError(f'The save_file_arg argument must be type dict, not {type(save_file_arg)}')
        if isinstance(inputs, list):
            for n, i in enumerate(node.input):
                v = tensors.make_ndarray_from_tensor(Graph.GetInputTensorFromFileArg(inputs[n], save_file_arg)) if n in file_args else inputs[n]
                graph_inputs.append(onnx.helper.make_tensor_value_info(i, onnx.helper.np_dtype_to_tensor_dtype(v.dtype), v.shape))
                node_inputs_list.append(tensors.make_tensor_from_ndarray(v)) 
        elif isinstance(inputs, dict):
            for name, val in inputs.iteritems():
                if name in file_args and not isinstance(val, str):
                    raise TypeError('The type of a file_arg node input must be str, not {type(val)}.')
                if not name in file_args and not isinstance(val, ndarray):
                    raise TypeError('The type of a node input must be ndarray, not {type(val)}.')
                v = tensors.make_ndarray_from_tensor(Graph.GetInputTensorFromFileArg(val, save_file_arg)) if name in file_args else val
                graph_inputs.append(onnx.helper.make_tensor_value_info(name, onnx.helper.np_dtype_to_tensor_dtype(v.dtype), v.shape))
                node_inputs_dict[name] = tensors.make_tensor_from_ndarray(v)
        else:
            raise TypeError(f'The type of inputs: {type(inputs)} is not supported.')
        
        graph_outputs = map(lambda i: onnx.helper.make_tensor_value_info(i, onnx.TensorProto.INT32, [0]), node.output)
        graph = onnx.helper.make_graph([node], node.name + "_graph", graph_inputs, graph_outputs)
        model = onnx.helper.make_model(graph)
        rep = prepare(model)
        return rep.run_graph_node(tensors.make_tensor_array(node_inputs_list), node.name) if isinstance(inputs, list) else rep.run_graph_node(tensors.make_tensor_dict(node_inputs_dict), node.name)
        
    @classmethod
    def prepare_file(cls, file_path:str) -> LokadOnnxRep:
        graph = load_graph(file_path)
        return LokadOnnxRep(graph)

    @classmethod
    def set_debug_mode(cls):
        Graph.SetDebugMode()


def load_graph(file_path:str) -> ComputationalGraph:
    return Graph.LoadFromFile(file_path)

prepare = LokadOnnxBackend.prepare

prepare_file = LokadOnnxBackend.prepare_file

run_node = LokadOnnxBackend.run_node

run_model = LokadOnnxBackend.run_model

supports_device = LokadOnnxBackend.supports_device

set_debug_mode = LokadOnnxBackend.set_debug_mode