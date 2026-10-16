"""Check operator rewrites against native ORT, independent of full weights."""
import importlib.util
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
import pytest

from common import session

path = Path(__file__).resolve().parents[3] / "eng/prepare-embeddinggemma2.py"
spec = importlib.util.spec_from_file_location("prepare_embeddinggemma2", path)
preparation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)


def value(name, shape, kind=onnx.TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, kind, shape)


def model(nodes, inputs, outputs, initializers=()):
    return helper.make_model(helper.make_graph(nodes, "rewrite-test", inputs, outputs, list(initializers)),
                             opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
                             ir_version=10)


def compare(tmp_path, original, feeds):
    prepared, _ = preparation.rewrite_model(original)
    source, target = tmp_path / "source.onnx", tmp_path / "prepared.onnx"
    onnx.save_model(original, source)
    onnx.save_model(prepared, target)
    expected = session(source).run(None, feeds)
    actual = session(target).run(None, feeds)
    assert len(expected) == len(actual)
    for x, y in zip(actual, expected):
        assert x.shape == y.shape and x.dtype == y.dtype
        if x.dtype == np.bool_:
            np.testing.assert_array_equal(x, y)
        else:
            np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("epsilon", [1e-6, 0.000512])
def test_rms_preserves_epsilon_and_does_not_subtract_mean(tmp_path, epsilon):
    node = helper.make_node("SimplifiedLayerNormalization", ["x", "scale"], ["y"], epsilon=epsilon, axis=-1, stash_type=1)
    original = model([node], [value("x", [2, 3, 8]), value("scale", [8])], [value("y", [2, 3, 8])])
    rng = np.random.default_rng(19)
    compare(tmp_path, original, {"x": (3 + rng.standard_normal((2, 3, 8))).astype(np.float32),
                                 "scale": rng.standard_normal(8).astype(np.float32)})


@pytest.mark.parametrize("bias", [False, True])
def test_skip_rms_preserves_residual_output_slot_three(tmp_path, bias):
    inputs = ["x", "skip", "scale"] + (["bias"] if bias else [])
    node = helper.make_node("SkipSimplifiedLayerNormalization", inputs, ["y", "", "", "sum"], domain="com.microsoft", epsilon=1e-6)
    descriptions = [value("x", [1, 3, 8]), value("skip", [1, 3, 8]), value("scale", [8])]
    if bias:
        descriptions.append(value("bias", [8]))
    original = model([node], descriptions, [value("y", [1, 3, 8]), value("sum", [1, 3, 8])])
    rng = np.random.default_rng(31)
    feeds = {"x": rng.standard_normal((1, 3, 8)).astype(np.float32),
             "skip": rng.standard_normal((1, 3, 8)).astype(np.float32), "scale": np.ones(8, np.float32)}
    if bias:
        feeds["bias"] = rng.standard_normal(8).astype(np.float32)
    compare(tmp_path, original, feeds)


@pytest.mark.parametrize("head_dim", [256, 512])
def test_rotary_layout_handles_multiple_heads_and_batches(tmp_path, head_dim):
    node = helper.make_node("RotaryEmbedding", ["x", "position", "cos", "sin"], ["y"], domain="com.microsoft", interleaved=0)
    original = model([node], [value("x", [2, 4, 2 * head_dim]), value("cos", [4, head_dim // 2]), value("sin", [4, head_dim // 2])],
                     [value("y", [2, 4, 2 * head_dim])], [numpy_helper.from_array(np.array([0], np.int64), "position")])
    rng = np.random.default_rng(47)
    angle = rng.standard_normal((4, head_dim // 2)).astype(np.float32)
    compare(tmp_path, original, {"x": rng.standard_normal((2, 4, 2 * head_dim)).astype(np.float32),
                                 "cos": np.cos(angle), "sin": np.sin(angle)})


def test_float_less_or_equal_keeps_ordered_nan_semantics(tmp_path):
    node = helper.make_node("LessOrEqual", ["x", "y"], ["z"])
    original = model([node], [value("x", [5]), value("y", [1])], [value("z", [5], onnx.TensorProto.BOOL)])
    compare(tmp_path, original, {"x": np.array([np.nan, -np.inf, -0.0, 0.0, np.inf], np.float32),
                                 "y": np.array([0], np.float32)})


def test_unqualified_normalization_axis_is_rejected():
    node = helper.make_node("SimplifiedLayerNormalization", ["x", "scale"], ["y"], axis=1)
    original = model([node], [value("x", [1, 3, 8]), value("scale", [8])], [value("y", [1, 3, 8])])
    with pytest.raises(ValueError, match="contract"):
        preparation.rewrite_model(original)
