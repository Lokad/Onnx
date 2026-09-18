"""Regenerate independent If fixtures: python generate_if_ort.py (onnx, numpy, onnxruntime).

Each record retains the actual protobuf and native result, so importer, optimizer,
capture binding and execution are exercised together. No Lokad code is used.
"""
from pathlib import Path
import base64
import hashlib
import json
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
import onnxruntime as ort


def vi(name, dims, dtype=T.FLOAT):
    return h.make_tensor_value_info(name, dtype, dims)


def branch(name, inputs, output, op="Identity", dims=(2,), init=()):
    return h.make_graph([h.make_node(op, inputs, [output], name=name)], name, [], [vi(output, dims)], list(init))


def choose(name, cond, then, otherwise, outputs):
    return h.make_node("If", [cond], outputs, name=name, then_branch=then, else_branch=otherwise)


records = []
def add(name, nodes, inputs, outputs, feeds, initializers=(), version=14):
    model = h.make_model(h.make_graph(nodes, name, inputs, outputs, list(initializers)),
                         opset_imports=[h.make_opsetid("", version)], ir_version=8)
    onnx.checker.check_model(model)
    raw = model.SerializeToString()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(raw, options, providers=["CPUExecutionProvider"])
    for take_then in (True, False):
        feed = dict(feeds)
        feed["cond"] = np.full(feeds["cond"].shape, take_then, dtype=np.bool_)
        values = session.run(None, feed)
        records.append(dict(name=name + ("-then" if take_then else "-else"), model=base64.b64encode(raw).decode(),
            sha256=hashlib.sha256(raw).hexdigest(), inputs=[dict(name=k, type=str(v.dtype), dims=list(v.shape), data=v.ravel().tolist()) for k,v in feed.items()],
            outputs=[dict(name=d.name, dims=list(v.shape), data=v.ravel().tolist()) for d,v in zip(session.get_outputs(),values)]))


x = np.array([-1, 8], np.float32)
feeds = dict(x=x, cond=np.array(True))
inputs = [vi("x", [2]), vi("cond", [], T.BOOL)]
then = branch("same", ["x"], "value")
otherwise = branch("opposite", ["x"], "value", "Neg")
add("basic", [choose("choose", "cond", then, otherwise, ["y"])], inputs, [vi("y", [2])], feeds)
add("vector-condition", [choose("choose", "cond", then, otherwise, ["y"])],
    [vi("x", [2]), vi("cond", [1], T.BOOL)], [vi("y", [2])], dict(x=x, cond=np.array([True])))

# One capture feeds two output nodes, and branch output ordering matters.
direct = h.make_graph([h.make_node("Identity", ["x"], ["copy"], name="copy"), h.make_node("Neg", ["x"], ["negative"], name="neg")], "direct", [], [vi("copy", [2]), vi("negative", [2])])
add("direct-multiple", [choose("choose", "cond", direct, direct, ["y", "z"])], inputs, [vi("y", [2]), vi("z", [2])], feeds)

# Different output sizes are legal starting at opset 11.
small = branch("small", ["local"], "value", init=[nh.from_array(np.array([9], np.float32), "local")], dims=(1,))
for version in (11, 14):
    add("different-shapes-v" + str(version), [choose("choose", "cond", then, small, ["y"])], inputs, [vi("y", [None])], feeds, version=version)

# Sibling branches reuse names, and only one branch shadows the outer x.
inner = choose("inner", "cond", then, otherwise, ["inside"])
nested = h.make_graph([inner], "nested", [], [vi("inside", [2])], [nh.from_array(np.array([7,9], np.float32), "x")])
add("nested-shadow", [choose("outer", "cond", nested, otherwise, ["y"])], inputs, [vi("y", [2])], feeds)

for producer in ("Add", "Conv"):
    dims = [1,1,2] if producer == "Conv" else [2]
    weight = np.ones([1,1,1], np.float32) if producer == "Conv" else np.zeros([2], np.float32)
    captured = branch("capture", ["pre"], "captured", dims=dims)
    nodes = [h.make_node(producer,["x","w"],["pre"],name="producer"), h.make_node("Relu",["pre"],["post"],name="relu"),
             choose("choose","cond",captured,captured,["y"])]
    add(producer.lower()+"-fusion-capture", nodes, [vi("x",dims),vi("cond",[],T.BOOL)], [vi("y",dims),vi("post",dims)],
        dict(x=x.reshape(dims),cond=np.array(True)),[nh.from_array(weight,"w")])

constant = nh.from_array(np.array([-1,8],np.float32))
captured = branch("capture",["second"],"value")
add("constant-capture", [h.make_node("Constant",[],["first"],name="first",value=constant),
    h.make_node("Constant",[],["second"],name="second",value=constant),h.make_node("Neg",["first"],["other"],name="other"),
    choose("choose","cond",captured,captured,["y"])], inputs, [vi("y",[2]),vi("other",[2])],feeds)

destination = Path(__file__).with_name("if-ort.json")
destination.write_text(json.dumps(dict(onnx=onnx.__version__, onnxruntime=ort.__version__, cases=records),indent=2)+"\n",encoding="utf-8")
print(f"Wrote {len(records)} native fixtures to {destination}")
