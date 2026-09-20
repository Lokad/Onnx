"""Independent complete input construction, hashes and scalar diagnostic reference."""
import hashlib,math
import numpy as np
from common import BANKS,read

def digest(arrays):
    h=hashlib.sha256()
    for array in arrays:h.update(np.asarray(array,dtype='<f4').tobytes())
    return h.hexdigest()

def inputs(origin,definition):
    folder=origin/'capture'/definition['source'];meta=read(folder/'capture.json');result=[]
    assert len(meta['nodes'])==25
    columns=np.arange(definition['width'])%384
    for i,node in enumerate(meta['nodes']):
        assert node['index']==i and node['block']==384 and node['outer']==definition['rows'] and node['has_bias'] is True
        prefix=f'{i:02}'
        def load(kind,rows):
            values=np.fromfile(folder/(prefix+'-'+kind+'.f32'),dtype='<f4');assert values.size==rows*384 and np.isfinite(values).all()
            return values.reshape(rows,384)[:,columns].copy()
        x=load('x',definition['rows']);scale=load('scale',1)[0];bias=load('bias',1)[0] if definition['bias'] else None
        expected=None if definition['diagnostic'] else load('y',definition['rows'])
        result.append((x,scale,bias,np.float32(node['epsilon']),expected))
    return result

def describe(origin,definition):
    nodes=inputs(origin,definition)
    return dict(name=definition['name'],repeats=definition['repeats'],diagnostic=definition['diagnostic'],width=definition['width'],rows=definition['rows'],
        input_sha256=digest(n[0] for n in nodes),scale_sha256=digest(n[1] for n in nodes),bias_sha256=digest(n[2] for n in nodes if n[2] is not None),
        output_sha256=None if definition['diagnostic'] else digest(n[4] for n in nodes))

def scalar(nodes):
    outputs=[]
    for x,scale,bias,epsilon,_ in nodes:
        output=np.empty_like(x)
        for row in range(x.shape[0]):
            mean=sum(float(v) for v in x[row])/x.shape[1]
            variance=sum((float(v)-mean)**2 for v in x[row])/x.shape[1]
            inv=1/math.sqrt(variance+float(epsilon))
            for col in range(x.shape[1]):output[row,col]=((float(x[row,col])-mean)*inv)*float(scale[col])+(0 if bias is None else float(bias[col]))
        outputs.append(output)
    return np.concatenate([v.ravel() for v in outputs])
