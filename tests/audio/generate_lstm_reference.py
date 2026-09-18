"""Regenerate the small CPU LSTM reference fixture (development only).

python -m pip install numpy==2.2.4 onnx==1.22.0 onnxruntime==1.29.0
python tests/audio/generate_lstm_reference.py

The managed tests use the checked-in JSON and need no Python/native dependency.
"""
from pathlib import Path
import json
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort


def run_case(name, direction="forward", **attributes):
    rng = np.random.default_rng(20260918)
    directions = 2 if direction == "bidirectional" else 1
    arrays = {
        "X": rng.uniform(-.8, .8, (4, 3, 2)).astype(np.float32),
        "W": rng.uniform(-.8, .8, (directions, 12, 2)).astype(np.float32),
        "R": rng.uniform(-.4, .4, (directions, 12, 3)).astype(np.float32),
        "B": rng.uniform(-.2, .2, (directions, 24)).astype(np.float32),
        "sequence_lens": np.array([4, 2, 0], dtype=np.int32),
        "initial_h": rng.uniform(-.4, .4, (directions, 3, 3)).astype(np.float32),
        "initial_c": rng.uniform(-.4, .4, (directions, 3, 3)).astype(np.float32),
        "P": rng.uniform(-.2, .2, (directions, 9)).astype(np.float32),
    }
    if name == "clip_preserves_large_cell":
        arrays["initial_c"].fill(10)
    attrs = dict(hidden_size=3, direction=direction, **attributes)
    node = helper.make_node("LSTM", list(arrays), ["Y", "Y_h", "Y_c"], **attrs)
    outputs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, directions, 3, 3]),
               helper.make_tensor_value_info("Y_h", TensorProto.FLOAT, [directions, 3, 3]),
               helper.make_tensor_value_info("Y_c", TensorProto.FLOAT, [directions, 3, 3])]
    graph = helper.make_graph([node], name, [], outputs,
                              [numpy_helper.from_array(v, k) for k,v in arrays.items()])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    onnx.checker.check_model(model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    values = session.run(None, {})
    return dict(name=name, attributes=attrs,
                inputs={k: dict(shape=list(v.shape), values=v.flatten().tolist()) for k,v in arrays.items()},
                outputs=[dict(shape=list(v.shape), values=v.flatten().tolist()) for v in values])


cases = [run_case(direction, direction) for direction in ("forward", "reverse", "bidirectional")]
cases += [run_case("coupled_peephole", "bidirectional", input_forget=1, clip=.25),
          run_case("clip_preserves_large_cell", clip=.1),
          run_case("alpha_after_unused_function", activations=["Sigmoid", "LeakyRelu", "Tanh"], activation_alpha=[.3]),
          run_case("independent_alpha_beta_lists", activations=["LeakyRelu", "HardSigmoid", "Affine"],
                   activation_alpha=[.15, .25, 1.1], activation_beta=[.4, -.05]),
          run_case("partial_lists_use_defaults", "bidirectional",
                   activations=["Sigmoid", "LeakyRelu", "Tanh", "HardSigmoid", "Tanh", "Tanh"], activation_alpha=[.3]),
          run_case("extra_parameters_ignored", activations=["Sigmoid", "LeakyRelu", "Tanh"], activation_alpha=[.3, 123., 456.], activation_beta=[789.])]
for name, alpha, beta in [("Affine", .7, .1), ("Relu", None, None), ("LeakyRelu", None, None),
                           ("ThresholdedRelu", .05, None), ("ScaledTanh", .9, .8), ("HardSigmoid", None, None),
                           ("Elu", None, None), ("Softsign", None, None), ("Softplus", None, None)]:
    args = {"activations": ["Sigmoid", name, name]}
    if alpha is not None: args["activation_alpha"] = [alpha, alpha]
    if beta is not None: args["activation_beta"] = [beta, beta]
    cases.append(run_case("activation_" + name, **args))
result = dict(provenance=dict(onnx=onnx.__version__, onnxruntime=ort.__version__, numpy=np.__version__,
                             opset=17, ir_version=8, provider="CPUExecutionProvider", threads=1,
                             execution="sequential", optimization="all", seed=20260918), cases=cases)
destination = Path(__file__).resolve().parents[1] / "Lokad.Onnx.Backend.Tests/fixtures/lstm-ort.json"
destination.parent.mkdir(exist_ok=True)
destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {len(cases)} cases to {destination}")
