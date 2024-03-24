import os
import onnx
import numpy as np

from interop import tensors, backend

file_dir = os.path.dirname(os.path.realpath(__file__))

onnx_model_file = os.path.join(file_dir, "..", "..", "tests", "Lokad.Onnx.Backend.Tests", "models", "mnist-8.onnx")
mnist4 = os.path.join(file_dir, "..", "..", "tests", "Lokad.Onnx.Backend.Tests", "images", "mnist4.png") + "::mnist"
node = onnx.helper.make_node(
            name="Add1",
            op_type="Add",
            inputs=["x", "y"],
            outputs=["Add1"],
)
x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.INT32, [4,5])
y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.INT32, [4,5])
Add1 = onnx.helper.make_tensor_value_info("Add1", onnx.TensorProto.INT32, [4,5])
graph = onnx.helper.make_graph([node], "graph1", [x, y], [Add1])
model = onnx.helper.make_model(graph)

def test_load_graph():
    g = backend.load_graph(os.path.join(file_dir, "..", "..", "tests", "Lokad.Onnx.Backend.Tests", "models", "mnist-8.onnx"))
    assert g.Nodes.Count == 12

def test_model_prepare():
    rep = backend.prepare_file(onnx_model_file)
    assert rep.graph.Nodes.Count == 12
    rep = backend.prepare(model)
    assert rep.graph.Nodes.Count == 1

def test_model_file_run():
    inputs = [mnist4]
    file_args = [0]
    rep = backend.prepare_file(onnx_model_file)
    r = rep.run(inputs, file_args=file_args)

def test_model_run():
    rep = backend.prepare(model)
    x = np.arange(0, 20).reshape(4, 5)
    r = rep.run([x, x])
    a = r['Add1']
    assert len(a.shape) == 2
    assert a.shape[1] == 5
    assert a[0,1] == 2  

def test_node_run():
    node = onnx.helper.make_node(
            name="Add2",
            op_type="Add",
            inputs=["x", "y"],
            outputs=["Add2"],
    )
    x = np.arange(0, 20).reshape(4, 5)
    y = np.arange(0, 20).reshape(4, 5)
    r = backend.run_node(node, [x, y])
    a = r['Add2']
    assert len(a.shape) == 2
    assert a.shape[1] == 5
    assert a[0,2] == 4  