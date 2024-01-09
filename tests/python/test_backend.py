import onnx
import pytest


from interop import backend

node = onnx.helper.make_node(
            "Squeeze",
            inputs=["x", "axes"],
            outputs=["y"],
)

def test_test():
    assert(True)

