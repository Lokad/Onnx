"""Regenerate small Clip fixtures, including the two voice-import regressions.

Development oracle: numpy==2.2.4, onnx==1.22.0, onnxruntime==1.29.0.
The managed tests consume checked-in bytes and have no native dependency.
"""
from pathlib import Path
import base64
import hashlib
import json
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

options = ort.SessionOptions()
options.intra_op_num_threads = options.inter_op_num_threads = 1
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
options.add_session_config_entry("session.intra_op.allow_spinning", "0")
options.add_session_config_entry("session.inter_op.allow_spinning", "0")
types = {"float32": (np.float32, TensorProto.FLOAT), "float64": (np.float64, TensorProto.DOUBLE),
         "int32": (np.int32, TensorProto.INT32), "int64": (np.int64, TensorProto.INT64)}
cases = []


def run(name, dtype, values, lo=None, hi=None, version=17, attrs=None):
    numpy_type, onnx_type = types[dtype]
    arrays = {"x": np.array(values, dtype=numpy_type)}
    names = ["x"]
    for key, bound in (("min", lo), ("max", hi)):
        if bound is None:
            names.append("")
        else:
            names.append(key)
            arrays[key] = np.array(bound, dtype=numpy_type)
    while names[-1] == "":
        names.pop()
    graph = helper.make_graph([helper.make_node("Clip", names, ["y"], **(attrs or {}))], name,
        [helper.make_tensor_value_info(key, onnx_type, list(value.shape)) for key, value in arrays.items()],
        [helper.make_tensor_value_info("y", onnx_type, list(arrays["x"].shape))])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", version)], ir_version=8)
    onnx.checker.check_model(model)
    raw_model = model.SerializeToString()
    result = ort.InferenceSession(raw_model, options, providers=["CPUExecutionProvider"]).run(None, arrays)[0]
    def tensor(value):
        return dict(shape=list(value.shape), bytes=value.tobytes().hex())
    cases.append(dict(name=name, dtype=dtype, model=base64.b64encode(raw_model).decode(),
                      model_sha256=hashlib.sha256(raw_model).hexdigest(),
                      inputs={key: tensor(value) for key, value in arrays.items()}, output=tensor(result)))


for dtype in ("float32", "float64"):
    values = [[-np.inf, -4, -0.0, 0.0], [1, 4, np.inf, np.nan]]
    run(dtype + "-default-finite", dtype, values)
    run(dtype + "-inverted-bounds", dtype, values, 3, 1)
    run(dtype + "-max-only", dtype, values, None, 2)
    run(dtype + "-rank-one-min", dtype, values, [1], None)
    run(dtype + "-equal-zero-bounds", dtype, values, -0.0, 0.0)
    run(dtype + "-nan-bounds", dtype, values, np.nan, np.nan)
    run(dtype + "-scalar", dtype, -7, -2, 2)
    run(dtype + "-empty", dtype, np.empty((0, 2)), -2, 2)
for dtype in ("int32", "int64"):
    info = np.iinfo(types[dtype][0])
    values = [[info.min, info.min + 1, -1, 0], [1, 2, info.max - 1, info.max]]
    run(dtype + "-default", dtype, values)
    run(dtype + "-inverted", dtype, values, 3, 1)
    run(dtype + "-max-only", dtype, values, None, info.max - 1)
    run(dtype + "-rank-one-min", dtype, values, [info.min + 1], None)
run("legacy-default", "float32", [-np.inf, -4, -0.0, 0.0, 4, np.inf, np.nan], version=6)
run("legacy-attributes", "float32", [[-4, 0], [1, 4]], version=10, attrs=dict(min=-1.5, max=2.25))
run("input-bound-v11", "float32", [-5, 1, 5], -2, 2, version=11)
destination = Path(__file__).resolve().parents[1] / "Lokad.Onnx.Backend.Tests/fixtures/clip-ort.json"
destination.write_text(json.dumps(dict(provenance=dict(onnx=onnx.__version__, onnxruntime=ort.__version__, numpy=np.__version__,
    ir_version=8, provider="CPUExecutionProvider", threads=1, execution="sequential", optimization="disabled",
    encoding="little-endian exact numeric bytes"), cases=cases), indent=2) + "\n", encoding="utf-8")
print("Wrote", len(cases), "Clip cases to", destination)
