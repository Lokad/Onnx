"""Native CPU fixtures for long InstanceNormalization rows; no model assets required."""
from pathlib import Path
import hashlib,json
import numpy as np
import onnx
from onnx import helper as h,TensorProto as T
import onnxruntime as ort

assert (np.__version__,onnx.__version__,ort.__version__)==('2.2.4','1.22.0','1.29.0')
options=ort.SessionOptions()
options.intra_op_num_threads=options.inter_op_num_threads=1
options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
model=h.make_model(h.make_graph([h.make_node('InstanceNormalization',['x','scale','bias'],['y'],epsilon=1e-5)],'long-rows',
    [h.make_tensor_value_info('x',T.FLOAT,[2,3,'width']),h.make_tensor_value_info('scale',T.FLOAT,[3]),h.make_tensor_value_info('bias',T.FLOAT,[3])],
    [h.make_tensor_value_info('y',T.FLOAT,[2,3,'width'])]),opset_imports=[h.make_opsetid('',17)],ir_version=8)
onnx.checker.check_model(model)
raw=model.SerializeToString()
session=ort.InferenceSession(raw,options,providers=['CPUExecutionProvider'])
scale=np.array([.75,-2.5,11],np.float32);bias=np.array([-.125,.5,100],np.float32)
def bits(v):return f'{int(np.float32(v).view(np.uint32)):08x}'
cases=[]
for width in (1,2,3,4,5,7,8,9,13,31,32,33,129,513,1773,5325,16000):
    for name,pattern in [('constant',[.1]),('periodic',[.1,-.2,.7,1.3,-2.1,.05,1/3]),('offset',[1024+i*.125 for i in range(7)])]:
        pattern=np.array(pattern,np.float32)
        x=np.empty((2,3,width),np.float32)
        for row in range(6):x.reshape(6,width)[row]=pattern[(np.arange(width)+row*5)%len(pattern)]
        y=session.run(None,dict(x=x,scale=scale,bias=bias))[0]
        maps=[]
        for source,expected in zip(x.reshape(6,width),y.reshape(6,width)):
            pairs={}
            for value in np.unique(source):
                outputs=expected[source==value].view(np.uint32)
                # Equal values in one row must use the same affine result,
                # including SIMD tails; retain every distinct native result.
                assert len(np.unique(outputs))==1
                pairs[bits(value)]=f'{int(outputs[0]):08x}'
            maps.append(pairs)
        cases.append(dict(name=f'{name}-{width}',width=width,pattern=[bits(v) for v in pattern],expected=maps,
                          input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),output_sha256=hashlib.sha256(y.tobytes()).hexdigest()))
out=Path(__file__).with_name('instance-normalization-long.json')
out.write_text(json.dumps(dict(onnx=onnx.__version__,onnxruntime=ort.__version__,numpy=np.__version__,
    build=ort.get_build_info(),model_sha256=hashlib.sha256(raw).hexdigest(),scale=[bits(v) for v in scale],bias=[bits(v) for v in bias],cases=cases),indent=2)+'\n',encoding='utf-8')
print(len(cases),'native cases')
