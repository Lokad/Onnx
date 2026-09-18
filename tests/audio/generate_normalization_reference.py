"""Pinned development oracle for the remaining segmentation operator slice.

Requires numpy==2.2.4, onnx==1.22.0, onnxruntime==1.29.0. No runtime dependency
is added to Lokad.Onnx; tests consume model and tensor bytes from the JSON.
"""
import base64
import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

assert (np.__version__, onnx.__version__, ort.__version__) == ("2.2.4", "1.22.0", "1.29.0"), "Use the pinned oracle versions"
rng = np.random.default_rng(20260918)
options = ort.SessionOptions()
options.intra_op_num_threads = options.inter_op_num_threads = 1
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
options.log_severity_level = 4
options.add_session_config_entry("session.intra_op.allow_spinning", "0")
options.add_session_config_entry("session.inter_op.allow_spinning", "0")
cases = []


def run(name, op, arrays, attrs=None, version=17):
    dtype = str(arrays["x"].dtype)
    onnx_type = TensorProto.FLOAT if dtype == "float32" else TensorProto.DOUBLE
    node = helper.make_node(op, list(arrays), ["y"], **(attrs or {}))
    graph = helper.make_graph([node], name,
        [helper.make_tensor_value_info(key, onnx_type, list(value.shape)) for key, value in arrays.items()],
        [helper.make_tensor_value_info("y", onnx_type, list(arrays["x"].shape))])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", version)], ir_version=8)
    onnx.checker.check_model(model)
    encoded = model.SerializeToString()
    result = ort.InferenceSession(encoded, options, providers=["CPUExecutionProvider"]).run(None, arrays)[0]
    def tensor(value): return dict(shape=list(value.shape), bytes=value.tobytes().hex())
    cases.append(dict(name=name, op=op, dtype=dtype, model=base64.b64encode(encoded).decode(),
                      model_sha256=hashlib.sha256(encoded).hexdigest(), inputs={key: tensor(value) for key, value in arrays.items()},
                      output=tensor(result)))


for shape in ((2, 3, 7), (1, 2, 3, 5), (2, 2, 2, 3, 4), (1, 2, 1), (0, 2, 3), (1, 0, 3), (1, 2, 0)):
    channels = shape[1]
    arrays = dict(x=rng.uniform(-2, 2, shape).astype(np.float32), scale=rng.uniform(-1.5, 1.5, channels).astype(np.float32),
                  bias=rng.uniform(-1, 1, channels).astype(np.float32))
    run("instance-" + "x".join(map(str, shape)), "InstanceNormalization", arrays)
for name, values, epsilon in (("constant", np.full((1, 2, 5), 4), 1e-5),
                               ("large-constant", np.full((1, 2, 5), 1e8), 1e-5),
                               ("offset", np.arange(14).reshape(1, 2, 7) / 8 + 1024, .03),
                               ("zero-epsilon", np.arange(14).reshape(1, 2, 7), 0),
                               ("special", np.array([[[np.inf, 0, 1], [np.nan, -1, 2]]]), 1e-5)):
    run("instance-" + name, "InstanceNormalization", dict(x=values.astype(np.float32), scale=np.array([.75, -1.25], np.float32),
                                                         bias=np.array([.25, 2], np.float32)), dict(epsilon=float(epsilon)))
audio = np.random.default_rng(731).normal(0, .15, (1, 1, 16000)).astype(np.float32)
run("instance-audio-16000", "InstanceNormalization", dict(x=audio, scale=np.array([1.25], np.float32), bias=np.array([-.2], np.float32)))
for dtype in (np.float32, np.float64):
    prefix = np.dtype(dtype).name
    values = rng.normal(0, 3, (2, 3, 4)).astype(dtype)
    for version in (11, 13, 17):
        for axis in (None, 0, 1, -1):
            run(f"log-{prefix}-v{version}-axis{axis}", "LogSoftmax", dict(x=values), {} if axis is None else dict(axis=axis), version)
    run("log-" + prefix + "-large", "LogSoftmax", dict(x=np.array([[1000, 0, -1000], [-1e20, 0, 1e20]], dtype)))
    run("log-" + prefix + "-empty", "LogSoftmax", dict(x=np.empty((2, 0, 3), dtype)), dict(axis=1))
    run("log-" + prefix + "-special", "LogSoftmax", dict(x=np.array([[np.inf, 0, 1], [-np.inf, -np.inf, -np.inf], [np.nan, 1, 2]], dtype)))
    for alpha in (None, .2, -.5, 0):
        run(f"leaky-{prefix}-{alpha}", "LeakyRelu", dict(x=np.array([[-np.inf, -4, -0.0, 0.0], [1, 4, np.inf, np.nan]], dtype)),
            {} if alpha is None else dict(alpha=float(alpha)))
    run("leaky-" + prefix + "-scalar", "LeakyRelu", dict(x=np.array(-2, dtype)), dict(alpha=.2))
    run("leaky-" + prefix + "-empty", "LeakyRelu", dict(x=np.empty((0, 2), dtype)))
path = Path(__file__).resolve().parents[1] / "Lokad.Onnx.Backend.Tests/fixtures/normalization-ort.json"
path.write_text(json.dumps(dict(provenance=dict(onnx=onnx.__version__, onnxruntime=ort.__version__, numpy=np.__version__,
    seed=20260918, ir_version=8, provider="CPUExecutionProvider", threads=1, execution="sequential", optimization="disabled",
    encoding="little-endian numeric bytes"), cases=cases), indent=2) + "\n", encoding="utf-8")
print("Wrote", len(cases), "normalization/activation cases to", path)
for case in cases:
    if case["name"] in ("instance-large-constant", "instance-offset"):
        print(case["name"], np.frombuffer(bytes.fromhex(case["output"]["bytes"]), dtype=np.float32).tolist())
