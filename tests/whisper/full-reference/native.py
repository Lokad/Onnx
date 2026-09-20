"""Double ORT sections, with independent scalar math.erf between sections."""
import math,gc
from common import np,packages
packages()
import onnxruntime as ort

def session(path):
    options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry('session.intra_op.allow_spinning','0');options.add_session_config_entry('session.inter_op.allow_spinning','0')
    options.log_severity_level=3
    return ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])

def double_erf(x):
    assert x.dtype==np.float64
    return np.fromiter((math.erf(float(v)) for v in x.flat),dtype=np.float64,count=x.size).reshape(x.shape)

def run(directory,stages,input_name,features,capture):
    values={input_name:features};records=[]
    for stage in stages:
        if stage['kind']=='ort':
            engine=session(directory/stage['file'])
            assert [i.name for i in engine.get_inputs()]==stage['inputs']
            assert [i.name for i in engine.get_outputs()]==stage['outputs']
            options=engine.get_session_options()
            observed=dict(intra=options.intra_op_num_threads,inter=options.inter_op_num_threads,
                execution=str(options.execution_mode),optimizations=str(options.graph_optimization_level),providers=engine.get_providers(),
                intra_spinning=options.get_session_config_entry('session.intra_op.allow_spinning'),inter_spinning=options.get_session_config_entry('session.inter_op.allow_spinning'))
            assert observed==dict(intra=1,inter=1,execution='ExecutionMode.ORT_SEQUENTIAL',optimizations='GraphOptimizationLevel.ORT_DISABLE_ALL',providers=['CPUExecutionProvider'],intra_spinning='0',inter_spinning='0')
            outputs=engine.run(None,{name:values[name] for name in stage['inputs']});del engine
        else:
            assert stage['kind']=='math_erf' and len(stage['inputs'])==len(stage['outputs'])==1
            observed=dict(erf='math.erf',dtype='float64')
            outputs=[double_erf(values[stage['inputs'][0]])]
        for name,value in zip(stage['outputs'],outputs):
            assert value.dtype==np.float64 and np.isfinite(value).all(),name
            capture(name,value)
            if name in stage['keep']:values[name]=value
        for name in list(values):
            if name not in stage['keep']:del values[name]
        records.append(dict(index=stage['index'],kind=stage['kind'],settings=observed,outputs=[dict(name=n,shape=list(v.shape)) for n,v in zip(stage['outputs'],outputs)]))
        del outputs,value
    assert not values
    return records
