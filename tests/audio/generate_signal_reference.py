"""Small complete ONNX/native and independent NumPy references for audio primitives."""
from pathlib import Path
import argparse
import base64
import hashlib
import json
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def tensor(values):
    return dict(dtype=str(values.dtype), shape=list(values.shape), values=values.ravel().tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    assert (np.__version__, onnx.__version__, ort.__version__) == ('2.2.4','1.22.0','1.29.0')
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.log_severity_level = 4
    cases = []

    def add(name, op, inputs, attributes, version, independent=None, source='native'):
        names = ['x' + str(i) if value is not None else '' for i, value in enumerate(inputs)]
        values = {key: value for key, value in zip(names, inputs) if value is not None}
        infos = [helper.make_tensor_value_info(key, numpy_helper.from_array(value).data_type, value.shape) for key, value in values.items()]
        model = helper.make_model(helper.make_graph([helper.make_node(op,names,['y'],**attributes)],name,infos,
                    [helper.make_tensor_value_info('y',numpy_helper.from_array(inputs[0]).data_type,None)]),opset_imports=[helper.make_opsetid('',version)])
        model.ir_version = 10
        if source == 'native':
            session = ort.InferenceSession(model.SerializeToString(), options, providers=['CPUExecutionProvider'])
            expected = session.run(None, values)[0]
        elif source == 'onnx-reference':
            expected = ReferenceEvaluator(model).run(None, values)[0]
        else:
            expected = independent.astype(inputs[0].dtype)
        if independent is not None:
            assert expected.shape == independent.shape
            difference = np.abs(expected.astype(np.float64) - independent.astype(np.float64)) / np.maximum(1,np.abs(independent.astype(np.float64)))
            assert difference.max(initial=0) <= (1e-4 if expected.dtype == np.float32 else 1e-10), (name,difference.max())
        assert np.isfinite(expected).all()
        model.graph.output[0].CopyFrom(helper.make_tensor_value_info('y', numpy_helper.from_array(expected).data_type, expected.shape))
        onnx.checker.check_model(model)
        model_bytes = model.SerializeToString()
        cases.append(dict(name=name, op=op, opset=version, attributes=attributes, source=source,
                          model=base64.b64encode(model_bytes).decode('ascii'), model_sha256=hashlib.sha256(model_bytes).hexdigest(),
                          inputs=[None if v is None else tensor(v) for v in inputs], expected=tensor(expected)))
        print(name, expected.shape, source, flush=True)

    # Native's older complex multi-frame pointer arithmetic is not a correctness
    # oracle, and its FFT reads a real window as complex values for complex input.
    # Check complex inputs against ONNX ReferenceEvaluator and NumPy together.
    for name,dtype,batch,size,n,step,components,one,windowed,source in (
        ('real-rank2',np.float32,2,23,8,3,0,1,True,'native'),
        ('real-rank3-double',np.float64,2,23,8,3,1,0,False,'native'),
        ('odd-prime',np.float64,1,19,7,4,1,1,True,'native'),
        ('nonpower',np.float32,1,27,10,6,0,0,False,'native'),
        ('audio512',np.float64,1,900,512,160,0,1,True,'native'),
        ('unit-window',np.float32,1,5,1,2,1,1,False,'native'),
        ('complex-one',np.float64,1,8,8,2,2,0,False,'native'),
        ('complex-window',np.float64,1,8,8,2,2,0,True,'onnx-reference'),
        ('complex-batches',np.float64,2,21,8,3,2,0,True,'onnx-reference'),
        ('complex-prime',np.float32,2,15,5,2,2,0,False,'onnx-reference')):
        shape=(batch,size) if components==0 else (batch,size,components)
        signal=(np.sin(np.arange(np.prod(shape),dtype=np.float64)*0.31)*0.8).astype(dtype).reshape(shape)
        window=(0.5-0.5*np.cos(2*np.pi*np.arange(n)/n)).astype(dtype) if windowed else None
        lengths=np.array(n,dtype=np.int64)
        framed=[]
        for b in range(batch):
            raw=signal[b] if components==0 else signal[b,:,0]
            if components==2: raw=raw.astype(np.complex128)+1j*signal[b,:,1]
            blocks=[]
            for start in range(0,size-n+1,step):
                weighted=raw[start:start+n].astype(np.complex128)*(window.astype(np.float64) if window is not None else 1)
                bins=np.fft.fft(weighted)
                if one: bins=bins[:n//2+1]
                blocks.append(np.stack((bins.real,bins.imag),axis=-1))
            framed.append(blocks)
        independent=np.array(framed).astype(dtype)
        add(name,'STFT',[signal,np.array(step,dtype=np.int64),window,None if windowed else lengths],{'onesided':one},17,independent,source)
    signal=np.arange(8,dtype=np.float64).reshape(1,8,1)
    bins=np.fft.rfft(signal[0,:,0])
    independent=np.stack((bins.real,bins.imag),axis=-1)[None,None,:,:]
    add('omitted-window-and-length','STFT',[signal,np.array(2,dtype=np.int64)],{},17,independent,'onnx-reference')

    for dtype in (np.float32,np.float64,np.int32,np.int64):
        values=np.array([1,-2,3,-4,5,-6],dtype=dtype).reshape(2,3)
        for version in (13,18):
            inputs=[values] if version==13 else [values,np.array([-1],dtype=np.int64)]
            attrs=dict(keepdims=0,axes=[-1]) if version==13 else dict(keepdims=0)
            add('square-'+str(np.dtype(dtype))+'-'+str(version),'ReduceSumSquare',inputs,attrs,version)
    add('square-noop','ReduceSumSquare',[np.array([-3,2],np.float32),np.array([],np.int64)],dict(noop_with_empty_axes=1),18)
    add('square-scalar','ReduceSumSquare',[np.array(-7,np.float64)],dict(keepdims=0),17)
    add('square-empty','ReduceSumSquare',[np.empty((2,0,3),np.float32)],dict(axes=[1],keepdims=0),17)
    add('square-duplicate-axes','ReduceSumSquare',[np.arange(6,dtype=np.float64).reshape(2,3),np.array([-1,1],np.int64)],dict(keepdims=1),18)
    add('square-int64-large','ReduceSumSquare',[np.array([94906265,-94906266,12345],np.int64)],dict(keepdims=0),17)
    add('square-int32-overflow','ReduceSumSquare',[np.array([50000,-50000,2147483647],np.int32)],dict(keepdims=0),17)
    add('square-int64-overflow','ReduceSumSquare',[np.array([4000000000,-4000000000,9223372036854775807],np.int64)],dict(keepdims=0),17)
    add('square-int32-noop-overflow','ReduceSumSquare',[np.array([50000,3],np.int32),np.array([],np.int64)],dict(noop_with_empty_axes=1),18)
    add('square-int64-noop-overflow','ReduceSumSquare',[np.array([4000000000,3],np.int64),np.array([],np.int64)],dict(noop_with_empty_axes=1),18)
    for length in (3,4,5,7,8,15,16,31,32,101,257):
        values=np.full((4,length),np.float32(-16.63553237915039),np.float32)
        add('sum-constant-'+str(length),'ReduceSum',[values,np.array([-1],np.int64)],dict(keepdims=0),13)
    for dtype in (np.float32,np.float64):
        values=np.array([0.125,0.5,1,2,10,1e-20,1e20],dtype=dtype)
        add('log-'+str(np.dtype(dtype)),'Log',[values],{},13,np.log(values))
    for dtype in (np.float32,np.float64,np.int32,np.int64):
        values=np.arange(12,dtype=dtype).reshape(3,4)
        add('reflect-'+str(np.dtype(dtype)),'Pad',[values,np.array([1,2,1,2],np.int64)],dict(mode='reflect'),17,
            np.pad(values,((1,1),(2,2)),mode='reflect'))
    values=np.arange(6,dtype=np.int64)
    add('reflect-crop','Pad',[values,np.array([-2,2],np.int64)],dict(mode='reflect'),17,np.pad(values[2:],(0,2),mode='reflect'))
    add('reflect-repeat','Pad',[np.array([1,3,8],np.float32),np.array([7,6],np.int64)],dict(mode='reflect'),17,
        np.pad(np.array([1,3,8],np.float32),(7,6),mode='reflect'),'numpy')
    add('reflect-singleton','Pad',[np.array([9],np.int64),np.array([2,3],np.int64)],dict(mode='reflect'),17,
        np.full(6,9,dtype=np.int64),'numpy')
    record=dict(numpy=np.__version__,onnx=onnx.__version__,onnxruntime=ort.__version__,
                generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=cases)
    args.output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__': main()
