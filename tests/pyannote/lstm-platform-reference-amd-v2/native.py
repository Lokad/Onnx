"""Fresh AMD ORT executions of all original captured LSTM nodes, without timing."""
from pathlib import Path
import sys
import numpy as np
import onnxruntime as ort
import psutil
from protocol import pin, read, save


def main():
    base=Path(__file__).resolve().parents[1];output=Path(sys.argv[1]);fixtures=base/'fixtures'
    own=psutil.Process();assert own.cpu_affinity()==[2] and ort.__version__=='1.29.0'
    captured=read(fixtures/'output/result.json');assert len(captured['calls'])==12
    reports=[];maximum=0.;values=0
    for ordinal,call in enumerate(captured['calls']):
        options=ort.SessionOptions();options.log_severity_level=4
        options.intra_op_num_threads=options.inter_op_num_threads=1
        options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        for key in ['session.intra_op.allow_spinning','session.inter_op.allow_spinning']:options.add_session_config_entry(key,'0')
        session=ort.InferenceSession(str(fixtures/'native'/(str(ordinal).zfill(2)+'.onnx')),options,providers=['CPUExecutionProvider'])
        assert session.get_providers()==['CPUExecutionProvider']
        inputs={name:np.fromfile(fixtures/'output'/item['file'],dtype='<f4').reshape(item['shape']) for name,item in zip(call['input_names'],call['inputs'],strict=True) if item is not None}
        before={name:a.tobytes() for name,a in inputs.items()}
        first=session.run(call['output_names'],inputs);held=[a.tobytes() for a in first];second=session.run(call['output_names'],inputs)
        assert all(a.tobytes()==b.tobytes()==h for a,b,h in zip(first,second,held,strict=True))
        assert all(a.tobytes()==before[name] for name,a in inputs.items())
        for slot,(a,item) in enumerate(zip(first,call['outputs'],strict=True)):
            selected=np.fromfile(fixtures/'output'/item['file'],dtype='<f4').reshape(item['shape'])
            assert a.dtype==selected.dtype==np.float32 and a.shape==selected.shape and np.isfinite(a).all() and np.isfinite(selected).all()
            error=float((np.abs(a.astype('float64')-selected.astype('float64'))/np.maximum(1.,np.abs(a.astype('float64')))).max(initial=0))
            file=output/(str(ordinal).zfill(2)+'-'+str(slot)+'.f32');a.astype('<f4').tofile(file)
            reports.append(dict(name=call['name'],index=call['index'],slot=slot,file=file.name,pin=pin(file),shape=list(a.shape),values=int(a.size),maximum=error,exact_repeat=True,input_unchanged=True))
            maximum=max(maximum,error);values+=int(a.size)
    save(output/'result.json',dict(passed=maximum<=1e-4,reports=reports,maximum=maximum,values=values,version=ort.__version__,pid=own.pid,no_performance_measurement=True))
    assert maximum<=1e-4


if __name__=='__main__':main()
