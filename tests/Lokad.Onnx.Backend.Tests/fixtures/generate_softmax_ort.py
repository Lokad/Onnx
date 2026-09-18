"""Independent CPU ORT softmax fixtures; run explicitly with the pinned versions below."""
from pathlib import Path
import base64
import hashlib
import json
import numpy as np
import onnx
from onnx import helper as h, TensorProto as T
import onnxruntime as ort

assert (np.__version__, onnx.__version__, ort.__version__) == ("2.2.4", "1.22.0", "1.29.0")
rng = np.random.default_rng(20260918)
options = ort.SessionOptions()
options.intra_op_num_threads = options.inter_op_num_threads = 1
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
options.log_severity_level = 4
options.add_session_config_entry("session.intra_op.allow_spinning", "0")
options.add_session_config_entry("session.inter_op.allow_spinning", "0")
cases = []


def run(name, x, mask=None, axis=-1, version=13):
    arrays = dict(x=x)
    if mask is not None:
        arrays["mask"] = mask
    nodes = [] if mask is None else [h.make_node("Add", ["x", "mask"], ["added"], name="add")]
    nodes.append(h.make_node("Softmax", ["x" if mask is None else "added"], ["y"], name="softmax", axis=axis))
    model = h.make_model(h.make_graph(nodes, name,
        [h.make_tensor_value_info(k,T.FLOAT,list(v.shape)) for k,v in arrays.items()],
        [h.make_tensor_value_info("y",T.FLOAT,list(x.shape))]),opset_imports=[h.make_opsetid("",version)],ir_version=8)
    onnx.checker.check_model(model)
    raw = model.SerializeToString()
    output = ort.InferenceSession(raw,options,providers=["CPUExecutionProvider"]).run(None,arrays)[0]
    def tensor(value): return dict(shape=list(value.shape), bytes=value.tobytes().hex())
    cases.append(dict(name=name,model=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest(),
                      inputs={k:tensor(v) for k,v in arrays.items()},output=tensor(output)))


for block in (1,7,8,9,30,128,512,1500):
    x = rng.normal(0,5,(3,block)).astype(np.float32)
    mask = np.where(np.arange(block)%3==0,0,-10000).astype(np.float32)
    run(f"plain-{block}",x)
    run(f"shared-mask-{block}",x,mask)
run("padded30-of128",rng.normal(0,2,(1,2,128)).astype(np.float32),np.array([0]*30+[-np.finfo(np.float32).max]*98,np.float32).reshape(1,1,128))
for version in (11,13):
    run(f"axis1-v{version}",rng.normal(0,2,(2,3,7)).astype(np.float32),axis=1,version=version)
    run(f"axis-negative-v{version}",rng.normal(0,2,(2,3,7)).astype(np.float32),axis=-2,version=version)
run("zero-rows",np.empty((0,9),np.float32))
run("zero-width",np.empty((2,0),np.float32))
run("full-mask-fallback",rng.normal(0,3,(3,9)).astype(np.float32),rng.normal(0,1,(3,9)).astype(np.float32))
for name,values in (("nan",[np.nan,-1,0,1,2,3,4,5,6]),("infinite",[np.inf,-1,0,1,2,3,4,5,6]),
                    ("all-negative-inf",[-np.inf]*9),("partial-negative-inf",[-np.inf,-1,0,1,2,3,4,5,6]),
                    ("large-offset",[1e6+i for i in range(9)])):
    run(name,np.array(values,np.float32).reshape(1,9))
run("all-masked",np.zeros((3,9),np.float32),np.full(9,-np.inf,np.float32))
path = Path(__file__).with_name("softmax-ort.json")
path.write_text(json.dumps(dict(onnx=onnx.__version__,onnxruntime=ort.__version__,cases=cases),indent=2)+"\n",encoding="utf-8")
print(f"Wrote {len(cases)} native softmax fixtures")
