"""The retired trial environment must not override the model's GELU attribute."""
import os
import subprocess

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from test_opfuzz_conformance import ATOL, RTOL, RUNNER, read_txt


@pytest.mark.parametrize("approximation", ["none", "tanh"])
@pytest.mark.parametrize("mode", ["scalar", "simd"])
def test_model_controls_gelu_approximation(approximation, mode, tmp_path):
    values = np.array([-3, -2, -1, 1, 2, 3], dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node("Gelu", ["x"], ["y"], approximate=approximation)],
        "explicit-gelu", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [6])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [6])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)], ir_version=8)
    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)
    input_path = tmp_path / "x.txt"
    input_path.write_text("float32 1 6\n-3 -2 -1 1 2 3\n")
    outputs = []
    for trial in [None, "1"]:
        environment = os.environ.copy()
        environment.pop("LOKAD_ONNX_GELU_TANH", None)
        if trial is not None:
            environment["LOKAD_ONNX_GELU_TANH"] = trial
        output_path = tmp_path / ("default" if trial is None else "retired-trial")
        output_path.mkdir()
        result = subprocess.run(
            ["dotnet", RUNNER, "--model", str(model_path), "--mode", mode,
             "--input", "x=" + str(input_path), "--outdir", str(output_path)],
            env=environment, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        outputs.append(output_path / "dotnet_y.txt")
    assert outputs[0].read_bytes() == outputs[1].read_bytes(), "retired trial changed model semantics"
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    reference = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    expected = reference.run(None, {"x": values})[0]
    np.testing.assert_allclose(read_txt(outputs[0]), expected, rtol=RTOL, atol=ATOL)
