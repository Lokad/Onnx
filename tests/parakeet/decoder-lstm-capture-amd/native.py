"""Native correctness on complete actual calls; fed-weight sessions are not performance baselines."""
import hashlib
import os
from pathlib import Path
import sys
import numpy as np
import onnxruntime as ort
import psutil
from protocol import pin, read, save, verify
from checks import array, scaled


def main():
    assert sys.platform=='linux' and not sys.flags.optimize and len(sys.argv)==2
    base=Path(sys.argv[1]).resolve();payload=verify(base);spec=read(base/'capture-spec.json')
    process=psutil.Process();assert process.cpu_affinity()==[2]
    flags={k:v for k,v in os.environ.items() if k.lower().startswith(('dotnet_','complus_','lokad_'))};assert not flags
    assert ort.__version__=='1.29.0' and np.__version__=='2.2.4'
    assert pin(Path(sys.executable))==payload['interpreter']
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    for kind in ['intra','inter']:options.add_session_config_entry('session.'+kind+'_op.allow_spinning','0')
    sessions=[ort.InferenceSession(str(base/n['file']),options,providers=['CPUExecutionProvider']) for n in spec['nodes']]
    for session,node in zip(sessions,spec['nodes'],strict=True):
        assert session.get_providers()==['CPUExecutionProvider'] and pin(base/node['file'])==node['model']
        assert [v.name for v in session.get_inputs()]==[n for n in node['input_names'] if n]
        assert [v.name for v in session.get_outputs()]==node['output_names']
        actual=session.get_session_options()
        assert actual.intra_op_num_threads==actual.inter_op_num_threads==1
        assert actual.execution_mode==ort.ExecutionMode.ORT_SEQUENTIAL and actual.graph_optimization_level==ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        assert all(actual.get_session_config_entry('session.'+kind+'_op.allow_spinning')=='0' for kind in ['intra','inter'])
    capture=read(base/'capture/output/result.json');cache={};held=[];rows=[];maximum=0.0
    def load(item):
        if item['file'] not in cache:cache[item['file']]=array(base/'capture/output',item).reshape(-1)
        return cache[item['file']].reshape(item['shape'])
    for call in capture['calls']:
        feeds={name:load(item) for name,item in zip(call['input_names'],call['inputs'],strict=True) if name}
        before={name:hashlib.sha256(v.tobytes()).hexdigest() for name,v in feeds.items()};first=None
        for repeat in [0,1]:
            outputs=sessions[call['index']].run(call['output_names'],feeds)
            if repeat:assert all(a.tobytes()==b.tobytes() for a,b in zip(first,outputs,strict=True))
            else:first=outputs
            for index,value in enumerate(outputs):
                assert value.dtype==np.float32 and np.isfinite(value).all()
                error,worst=scaled(load(call['outputs'][index]),value);assert error<=1e-4
                maximum=max(maximum,error);raw=value.tobytes();digest=hashlib.sha256(raw).hexdigest()
                filename=f'{len(rows):04}.f32';path=base/'native'/filename
                with path.open('xb') as stream:stream.write(raw)
                held.append((value,digest));rows.append(dict(name=call['name'],step=call['step'],index=call['index'],repeat=repeat,output=index,
                    array=dict(file=filename,shape=list(value.shape),values=value.size,**pin(path)),max_error=error,worst_index=worst))
            assert all(hashlib.sha256(v.tobytes()).hexdigest()==before[name] for name,v in feeds.items())
        if call['index']==1:print(call['name'],call['step'],'native complete calls passed',flush=True)
    assert all(hashlib.sha256(v.tobytes()).hexdigest()==digest for v,digest in held)
    libraries={}
    for entry in process.memory_maps():
        path=Path(entry.path)
        if 'onnxruntime' in str(path) and '.so' in path.name and path.is_file():libraries[str(path)]=pin(path)
    assert libraries and all(payload['external'].get(name)==wanted for name,wanted in libraries.items())
    result=dict(passed=True,pid=process.pid,affinity=process.cpu_affinity(),onnxruntime=ort.__version__,numpy=np.__version__,
        providers=['CPUExecutionProvider'],settings=dict(intra=1,inter=1,sequential=True,all_optimizations=True,spinning=False),
        capture=pin(base/'capture/output/result.json'),spec=pin(base/'capture-spec.json'),models={n['file']:n['model'] for n in spec['nodes']},
        libraries=libraries,interpreter=pin(Path(sys.executable)),flags=flags,rows=rows,max_error=maximum,
        held_outputs_unchanged=True,inputs_unchanged=True,no_performance_measurement=True)
    save(base/'native/result.json',result)


if __name__=='__main__':main()
