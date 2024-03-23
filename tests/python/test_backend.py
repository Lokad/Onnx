import os
import onnx
import numpy as np

from interop import tensors, backend

file_dir = os.path.dirname(os.path.realpath(__file__))

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

def test_backend():
    onnx_model_file = os.path.join(file_dir, "..", "..", "tests", "Lokad.Onnx.Backend.Tests", "models", "mnist-8.onnx")
    rep = backend.prepare_file(onnx_model_file)
    assert rep.graph.Nodes.Count == 12
    rep = backend.prepare(model)
    assert rep.graph.Nodes.Count == 1
    x = np.arange(0, 20).reshape(4, 5)
    r = rep.run([x, x])

