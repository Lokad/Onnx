"""Both saved feature sources through the complete selected native encoder trace."""
from pathlib import Path
import argparse,collections,hashlib,json,os,sys
import numpy as np
import onnx,onnxruntime as ort,psutil
from common import ROOT,KINDS,pin,read,verify

def identity():
    package=Path(ort.__file__).parent
    files={str(p.resolve()):pin(p) for p in sorted(package.rglob('*')) if p.is_file() and p.suffix in ['.py','.pyd','.dll']}
    return dict(python=sys.version,executable=str(Path(sys.executable).resolve()),numpy=np.__version__,onnxruntime=ort.__version__,files=files)
def raw(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
def load(record):
    path=ROOT/record['file'];value=np.load(path,allow_pickle=False) if record['format']=='npy' else np.fromfile(path,dtype='<f4').reshape(record['shape'])
    assert value.dtype==np.float32 and list(value.shape)==record['shape'] and np.isfinite(value).all() and raw(value)==record['raw_sha256']
    return value

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--request-index',type=int,required=True);args=parser.parse_args()
    spec=read(args.manifest);verify(spec);assert identity()==spec['native_runtime']
    assert args.request_index in range(4) and not args.output.exists()
    process=psutil.Process();assert process.cpu_affinity()==[2] and (np.__version__,ort.__version__)==('2.2.4','1.29.0')
    flags={k:v for k,v in os.environ.items() if k.lower().startswith(('lokad_','dotnet_','complus_'))};assert flags=={}
    args.output.mkdir();options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry('session.intra_op.allow_spinning','0');options.add_session_config_entry('session.inter_op.allow_spinning','0')
    options.optimized_model_filepath=str(args.output/'optimized.onnx')
    options.add_session_config_entry('session.optimized_model_external_initializers_file_name','optimized.weights')
    options.add_session_config_entry('session.optimized_model_external_initializers_min_size_in_bytes','1024')
    session=ort.InferenceSession(str(ROOT/spec['model']),options,providers=['CPUExecutionProvider'])
    names=[row['name'] for row in spec['outputs']]
    assert session.get_providers()==['CPUExecutionProvider'] and [v.name for v in session.get_inputs()]==['input_features']
    assert [v.name for v in session.get_outputs()]==names
    optimized=onnx.load(args.output/'optimized.onnx',load_external_data=False)
    assert (args.output/'optimized.weights').is_file()
    assert all({e.key:e.value for e in v.external_data}['location']=='optimized.weights' for v in optimized.graph.initializer if v.external_data)
    census=dict(collections.Counter((node.domain+'::' if node.domain else '')+node.op_type for node in optimized.graph.node))
    optimized_files={name:pin(args.output/name) for name in ['optimized.onnx','optimized.weights']}
    modules={str(Path(m.path).resolve()):pin(Path(m.path)) for m in process.memory_maps() if 'onnxruntime' in m.path.lower() and Path(m.path).suffix in ['.dll','.pyd']}
    assert modules and all(spec['native_runtime']['files'].get(k)==v for k,v in modules.items())
    result=dict(schema=1,engine='native',request_index=args.request_index,complete=False,manifest_sha256=pin(args.manifest)['sha256'],native_runtime=identity(),modules=modules,
        affinity=process.cpu_affinity(),flags=flags,threads=1,sequential=True,all_optimizations=True,spinning=False,records=[],held_outputs={},
        optimized_nodes=len(optimized.graph.node),node_census=census,optimized_files=optimized_files,serialization=spec['native_serialization'])
    def save():
        temp=args.output/'result.tmp';temp.write_text(json.dumps(result,indent=2));temp.replace(args.output/'result.json')
    held={};save();item=spec['requests'][args.request_index]
    for kind in KINDS['native']:
        features=load(item['native_features' if kind=='NN' else 'managed_features']);before=raw(features)
        values=session.run(names,{'input_features':features});assert len(values)==41 and raw(features)==before
        outputs=[]
        for index,(value,description) in enumerate(zip(values,spec['outputs'])):
            assert value.dtype==np.float32 and list(value.shape)==description['shape'] and np.isfinite(value).all()
            filename=f'{kind}-{index:02}.f32';path=args.output/filename
            with path.open('xb') as stream:stream.write(value.tobytes())
            key=kind+':'+str(index);held[key]=value;result['held_outputs'][key]=raw(value)
            outputs.append(dict(index=index,name=description['name'],shape=list(value.shape),file=filename,sha256=raw(value),values=value.size))
        assert all(raw(v)==result['held_outputs'][key] for key,v in held.items())
        reference=load(item['baselines'][kind]);final=values[-1]
        scaled=np.abs(final.astype(np.float64)-reference.astype(np.float64))/np.maximum(1,np.abs(reference.astype(np.float64)))
        result['records'].append(dict(kind=kind,request=args.request_index,name=item['name'],input_sha256=before,inputs_unchanged=True,held_outputs_unchanged=True,outputs=outputs,
            instrumentation=dict(baseline_sha256=raw(reference),final_sha256=raw(final),bitwise=raw(reference)==raw(final),max_scaled=float(scaled.max()),failed_values=int((scaled>1e-4).sum()))))
        save();print('Completed trace',args.request_index,kind,'all 41 outputs',flush=True)
    assert all(raw(v)==result['held_outputs'][key] for key,v in held.items());verify(spec);assert identity()==spec['native_runtime']
    result['complete']=True;save()

if __name__=='__main__':main()
