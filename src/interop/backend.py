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
    def __init__(self, model:onnx.ModelProto, graph:ComputationalGraph):
        super().__init__()
        self.model = model
        self.graph = graph

    def run(self, inputs, **kwargs):
        file_args = kwargs['file_args'] if 'file_args' in kwargs else []
        save_file_arg = kwargs['save_file_arg'] if 'save_file_arg' in kwargs else False
        use_initializers = kwargs['use_initializers'] if 'use_initializers' in kwargs else False
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
            r = self.graph.Execute(tensors.make_tensor_dictionary(_inputs), use_initializers)
        elif isinstance(inputs, list):
            _inputs=[]
            for index,value in enumerate(inputs):
                if index in file_args:
                    _inputs.append(Graph.GetInputTensorFromFileArg(value, save_file_arg))
                else:
                    _inputs.append(tensors.make_tensor_from_ndarray(value))
            r = self.graph.Execute(tensors.make_tensor_array(_inputs), use_initializers)
        else:
            raise RuntimeError(f'The input type {type(inputs)} is not supported by the backend.')
        
        if not r:
            raise RuntimeError('The graph did not execute successfully.')
        outputs = self.convert_graph_outputs(self.graph.Outputs, 'Outputs')
        self.graph.Reset()
        return outputs
    
    def run_node(self, node:Any, inputs:Any, **kwargs: Dict[str, Any]):
        node_name = '' 
        if isinstance(node, onnx.NodeProto):
           node_name = node.name
        elif isinstance(node, str):
              node_name = node
        else: raise TypeError(f'node must be type NodeProto or str, not {type(node)}.')
        file_args = kwargs['file_args'] if 'file_args' in kwargs else []
        save_file_arg = kwargs['save_file_arg'] if 'save_file_arg' in kwargs else False
        use_initializers = kwargs['use_initializers'] if 'use_initializers' in kwargs else False

        r = False

        if isinstance(inputs, list):
            _inputs=[]
            for n, i in enumerate(inputs):
                v = Graph.GetInputTensorFromFileArg(inputs[n], save_file_arg) if n in file_args else tensors.make_tensor_from_ndarray(inputs[n])
                _inputs.append(v) 
            r = self.graph.ExecuteNode(tensors.make_tensor_array(_inputs), node_name, use_initializers)
        elif isinstance(inputs, dict):
            _inputs={}
            for name, val in inputs.iteritems():
                if name in file_args and not isinstance(val, str):
                    raise TypeError('The type of a file_arg node input must be str, not {type(val)}.')
                if not name in file_args and not isinstance(val, np.ndarray):
                    raise TypeError('The type of a node input must be ndarray, not {type(val)}.')
                v = Graph.GetInputTensorFromFileArg(val, save_file_arg) if name in file_args else tensors.make_tensor_from_ndarray(val)
                _inputs[name] = v
            r = self.graph.ExecuteNode(tensors.make_tensor_dict(_inputs), node_name, use_initializers)
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
    
    def get_onnx_node(self, name:str):
        for node in self.model.graph.node:
            if node.name == name:
                return node
        raise RuntimeError(f'The graph does not contain the node {name}.')
    
    def get_initializer(self, name:str):
        if self.graph.Initializers.ContainsKey(name):
            return tensors.make_ndarray_from_tensor(self.graph.Initializers[name])
        else:
            raise ValueError(f'The graph does not contain the initializer {name}.')

    def get_input_ndarray_from_file_arg(self, arg:str, save_input=False):
        i = Graph.GetInputTensorFromFileArg(arg, save_input)
        return tensors.make_ndarray_from_tensor(i) if i != None else None
    
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
        graph = load_graph(model.SerializeToString())
        return LokadOnnxRep(model, graph)
    
    @classmethod
    def run_node(cls, node: onnx.NodeProto, inputs: Any, device: str = "CPU", outputs_info: Optional[Sequence[Tuple[np.dtype, Tuple[int, ...]]]] = None,**kwargs: Dict[str, Any],) -> Optional[Tuple[Any, ...]]:
        super(LokadOnnxBackend, cls).run_node(node, inputs, device, outputs_info)
    
        if node.name == None or node.name == '':
            node.name = 'node1'
        file_args = kwargs['file_args'] if 'file_args' in kwargs else []
        save_file_arg = kwargs['save_file_arg'] if 'save_file_arg' in kwargs else False
        if not isinstance(file_args, list):
            raise TypeError(f'The file_args argument must be type list, not {type(file_args)}')
        if not isinstance(save_file_arg, bool):
            raise TypeError(f'The save_file_arg argument must be type dict, not {type(save_file_arg)}')
        
        graph_inputs = []
        graph_outputs = []
        node_inputs_list = []
        node_inputs_dict = {}

        if isinstance(inputs, list):
            for n, i in enumerate(node.input):
                v = tensors.make_ndarray_from_tensor(Graph.GetInputTensorFromFileArg(inputs[n], save_file_arg)) if n in file_args else inputs[n]
                graph_inputs.append(onnx.helper.make_tensor_value_info(i, onnx.helper.np_dtype_to_tensor_dtype(v.dtype), v.shape))
                node_inputs_list.append(v) 
        elif isinstance(inputs, dict):
            for name, val in inputs.iteritems():
                if name in file_args and not isinstance(val, str):
                    raise TypeError('The type of a file_arg node input must be str, not {type(val)}.')
                if not name in file_args and not isinstance(val, np.ndarray):
                    raise TypeError('The type of a node input must be ndarray, not {type(val)}.')
                v = tensors.make_ndarray_from_tensor(Graph.GetInputTensorFromFileArg(val, save_file_arg)) if name in file_args else val
                graph_inputs.append(onnx.helper.make_tensor_value_info(name, onnx.helper.np_dtype_to_tensor_dtype(v.dtype), v.shape))
                node_inputs_dict[name] = v
        else:
            raise TypeError(f'The inputs type {type(inputs)} is not supported by the backend.')
        
        graph_outputs = map(lambda i: onnx.helper.make_tensor_value_info(i, onnx.TensorProto.INT32, [0]), node.output)
        graph = onnx.helper.make_graph([node], node.name + "_graph", graph_inputs, graph_outputs)
        model = onnx.helper.make_model(graph)
        rep = prepare(model)
        return rep.run_node(node.name, node_inputs_list, **kwargs) if isinstance(inputs, list) else rep.run_node(node.name, node_inputs_dict, **kwargs)
        
    @classmethod
    def prepare_file(cls, file_path:str) -> LokadOnnxRep:
        model = onnx.load(file_path)
        graph = load_graph(file_path)
        return LokadOnnxRep(model, graph)

    @classmethod
    def set_debug_mode(cls):
        Graph.SetDebugMode()

    @classmethod
    def get_input_ndarray_from_file_arg(cls, arg:str, save_input=False):
        i = Graph.GetInputTensorFromFileArg(arg, save_input)
        return tensors.make_ndarray_from_tensor(i) if i != None else None
    
    @classmethod
    def load_graph(cls, model) -> ComputationalGraph:
        return Graph.Load(model)

prepare = LokadOnnxBackend.prepare

prepare_file = LokadOnnxBackend.prepare_file

run_node = LokadOnnxBackend.run_node

run_model = LokadOnnxBackend.run_model

supports_device = LokadOnnxBackend.supports_device

set_debug_mode = LokadOnnxBackend.set_debug_mode

get_input_ndarray_from_file_arg = LokadOnnxBackend.get_input_ndarray_from_file_arg

load_graph = LokadOnnxBackend.load_graph