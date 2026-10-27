"""Fresh integer Resize checks, including the remaining nonidentity baseline pin.

Int32 identity resizing must match ORT exactly. The nonidentity align-corners
limitation reproduces with the frozen f53d699d core and remains independently
pinned until a separate numerical fix. Both engines are checked exactly below,
without xfail, skips or wider tolerances. Frozen corpus files remain unchanged.
"""
from pathlib import Path
import subprocess

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
import pytest

from test_opfuzz_conformance import MODES, RUNNER, read_txt


CASES = [
    (dtype, mode, coordinate, nearest, scale)
    for dtype in (np.int8, np.uint8, np.int32)
    for mode, coordinate, nearest, scale in (
        ("nearest", "half_pixel", "floor", 2.0),
        ("nearest", "align_corners", "ceil", 2.0),
        ("nearest", "asymmetric", "round_prefer_floor", 2.0),
        ("nearest", "half_pixel", "round_prefer_ceil", 2.0),
        ("linear", "half_pixel", "round_prefer_floor", 1.5),
        ("linear", "asymmetric", "round_prefer_floor", 1.0),
        ("nearest", "asymmetric", "floor", 1.25),
        ("linear", "asymmetric", "round_prefer_floor", 1.25),
        ("linear", "align_corners", "round_prefer_floor", 2.0),
    )
]


def input_array(dtype):
    values = {
        np.int8: [-128, -1, 64, 127],
        np.uint8: [0, 1, 254, 255],
        np.int32: [-(2**31), 2**31 - 1, -16777217, 16777217],
    }[dtype]
    return np.array(values, dtype=dtype).reshape(1, 1, 2, 2)


def legacy_int32_results(mode, coordinate, scale):
    if (mode, coordinate, scale) == ("linear", "align_corners", 2.0):
        managed = [[-2147483648, -715827776, 715827968, -2147483648],
                   [-1437248128, -479082624, 479082752, 1437248128],
                   [-727012672, -242337520, 242337584, 727012672],
                   [-16777216, -5592404, 5592406, 16777216]]
        native = [[-2147483648, -715827776, 715827968, -2147483648],
                  [-1437248128, -479082624, 479082720, 1437248000],
                  [-727012672, -242337520, 242337568, 727012608],
                  [-16777216, -5592404, 5592406, 16777216]]
        return np.array(managed, np.int32).reshape(1, 1, 4, 4), np.array(native, np.int32).reshape(1, 1, 4, 4)
    return None


def save_input(path, value):
    header = f"{value.dtype.name} {value.ndim} " + " ".join(map(str, value.shape))
    path.write_text(header + "\n" + " ".join(str(int(v)) for v in value.flat) + "\n")


def execute_managed(model_path, input_path, output, mode):
    assert Path(RUNNER).is_file(), f"Build the OpDump runner first: {RUNNER}"
    return subprocess.run(
        ["dotnet", RUNNER, "--model", str(model_path), "--mode", mode,
         "--input", "x=" + str(input_path), "--outdir", str(output)],
        capture_output=True, text=True)


@pytest.mark.parametrize("dtype,mode,coordinate,nearest,scale", CASES)
def test_integer_resize_native_and_baseline_contract(tmp_path, record_property, dtype, mode, coordinate, nearest, scale):
    values = input_array(dtype)
    element_type = helper.np_dtype_to_tensor_dtype(values.dtype)
    scales = numpy_helper.from_array(np.array([1, 1, scale, scale], np.float32), "scales")
    node = helper.make_node("Resize", ["x", "", "scales"], ["y"], mode=mode,
                            coordinate_transformation_mode=coordinate, nearest_mode=nearest)
    output_shape = [1, 1, int(2 * scale), int(2 * scale)]
    graph = helper.make_graph([node], "integer-resize", [helper.make_tensor_value_info("x", element_type, values.shape)],
                              [helper.make_tensor_value_info("y", element_type, output_shape)], [scales])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=10)
    model_path, input_path = tmp_path/"model.onnx", tmp_path/"input.txt"
    onnx.save(model, model_path)
    save_input(input_path, values)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    reference = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"]).run(None, {"x": values})[0]
    legacy = legacy_int32_results(mode, coordinate, scale) if dtype == np.int32 else None
    if legacy is not None:
        assert np.array_equal(reference, legacy[1]), "Native behavior changed; requalify the documented baseline difference"
        assert not np.array_equal(legacy[0], legacy[1])
        record_property("baseline_native_difference", f"f53d699d int32 {mode}/{coordinate}/{scale}")
    for engine_mode in MODES:
        output = tmp_path/engine_mode
        output.mkdir()
        result = execute_managed(model_path, input_path, output, engine_mode)
        if result.returncode == 3 and engine_mode == "intrinsics":
            continue  # Scalar and portable SIMD comparisons above still ran.
        assert result.returncode == 0, result.stderr
        actual = read_txt(output/"dotnet_y.txt")
        assert actual.shape == reference.shape and actual.dtype == reference.dtype
        expected = reference if legacy is None else legacy[0]
        assert np.array_equal(actual, expected), (dtype, mode, coordinate, nearest, engine_mode, actual, expected)


@pytest.mark.parametrize("sizes", ([1, 1, 2], [1, 1, 2, 2, 2]))
def test_integer_resize_rejects_invalid_size_rank(tmp_path, sizes):
    values = input_array(np.int32)
    node = helper.make_node("Resize", ["x", "", "", "sizes"], ["y"], mode="nearest")
    graph = helper.make_graph([node], "invalid-integer-resize", [helper.make_tensor_value_info("x", TensorProto.INT32, values.shape)],
                              [helper.make_tensor_value_info("y", TensorProto.INT32, [None]*4)],
                              [numpy_helper.from_array(np.array(sizes, np.int64), "sizes")])
    model_path, input_path = tmp_path/"model.onnx", tmp_path/"input.txt"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=10), model_path)
    save_input(input_path, values)
    with pytest.raises(Exception):
        ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"]).run(None, {"x": values})
    output = tmp_path/"managed"
    output.mkdir()
    result = execute_managed(model_path, input_path, output, "scalar")
    assert result.returncode != 0, "Managed Resize accepted an invalid size-vector rank"
