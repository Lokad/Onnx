"""Native encoder baseline bridges and managed-feature cross-input observations."""
from pathlib import Path
import argparse,hashlib,json,os,sys
import numpy as np
import onnxruntime as ort
import psutil
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"input-cross"))
from common import ROOT,pin,read,verify

def identity():
    package=Path(ort.__file__).parent
    files={str(p.resolve()):pin(p) for p in sorted(package.rglob('*')) if p.is_file() and p.suffix in ['.py','.pyd','.dll']}
    return dict(python=sys.version,executable=str(Path(sys.executable).resolve()),numpy=np.__version__,onnxruntime=ort.__version__,files=files)

def raw(v):return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
def load(record):
    p=ROOT/record['file'];v=np.load(p,allow_pickle=False) if record['format']=='npy' else np.fromfile(p,dtype='<f4').reshape(record['shape'])
    assert v.dtype==np.float32 and list(v.shape)==record['shape'] and np.isfinite(v).all() and raw(v)==record['raw_sha256'];return v

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--identity',action='store_true');parser.add_argument('--manifest',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--request-index',type=int);args=parser.parse_args()
    if args.identity:print(json.dumps(identity()));return
    assert args.manifest and args.output and not args.output.exists()
    spec=read(args.manifest);verify(spec);assert identity()==spec['native_runtime']
    assert spec['protocol']=='whisper-input-cross-isolated-case-v3' and args.request_index in range(21)
    process=psutil.Process();assert process.cpu_affinity()==[2] and (np.__version__,ort.__version__)==('2.2.4','1.29.0')
    flags={k:v for k,v in os.environ.items() if k.lower().startswith(('lokad_','dotnet_','complus_'))};assert flags=={}
    args.output.mkdir();options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry('session.intra_op.allow_spinning','0');options.add_session_config_entry('session.inter_op.allow_spinning','0')
    session=ort.InferenceSession(str(ROOT/spec['model']),options,providers=['CPUExecutionProvider'])
    assert session.get_providers()==['CPUExecutionProvider'] and [i.name for i in session.get_inputs()]==['input_features']
    modules={str(Path(m.path).resolve()):pin(Path(m.path)) for m in process.memory_maps() if 'onnxruntime' in m.path.lower() and Path(m.path).suffix in ['.dll','.pyd']}
    assert modules and all(spec['native_runtime']['files'].get(k)==v for k,v in modules.items())
    result=dict(schema=1,engine='native',request_index=args.request_index,complete=False,manifest_sha256=pin(args.manifest)['sha256'],native_runtime=identity(),modules=modules,
                affinity=process.cpu_affinity(),flags=flags,threads=1,sequential=True,all_optimizations=True,spinning=False,records=[],held_outputs={})
    def save():
        temp=args.output/'result.tmp';temp.write_text(json.dumps(result,indent=2));temp.replace(args.output/'result.json')
    held={};save()
    for index in [args.request_index]:
        item=spec['requests'][index]
        for kind,key in [('NN','native_features'),('NM','managed_features')]:
            values=load(item[key]);before=raw(values);output=session.run(['last_hidden_state'],{'input_features':values})[0]
            assert output.dtype==np.float32 and output.shape==(1,1500,1280) and np.isfinite(output).all() and raw(values)==before
            expected=load(item['native_hidden']);error=np.abs(output.astype(np.float64)-expected)/np.maximum(1,np.abs(expected.astype(np.float64)))
            name=f"{index:02}-{item['name']}-{kind}.f32";path=args.output/name
            with path.open('xb') as f:f.write(output.tobytes())
            baseline=raw(output)==item['native_hidden']['raw_sha256'] if kind=='NN' else None
            held[kind]=output;result['held_outputs'][kind]=raw(output)
            assert all(raw(v)==result['held_outputs'][k] for k,v in held.items())
            failed=int((error>1e-4).sum())
            result['records'].append(dict(request=index,name=item['name'],kind=kind,file=name,sha256=pin(path)['sha256'],shape=list(output.shape),input_sha256=before,
                baseline_matches=baseline,inputs_unchanged=True,held_outputs_unchanged=True,values=output.size,failed_values=failed,max_scaled=float(error.max()),numerical_passed=failed==0))
            save();assert baseline is not False,'Original native baseline bridge failed'
            print(index,item['name'],kind,'complete; numerical',failed==0,'baseline',baseline,flush=True)
    verify(spec);assert identity()==spec['native_runtime'];result['complete']=True;save()

if __name__=='__main__':main()
