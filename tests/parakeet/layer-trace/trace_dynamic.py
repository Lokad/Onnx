"""Correct the diagnosed static-shape instrumentation without replaying controls.

Four new native calls only: two feature inputs and their exact repeats. The old
managed traces and unmodified-model controls remain byte-bound reused evidence.
"""
from common import *
from native import outputs
from analyze import metrics
import copy
import shutil
import subprocess
import time
import traceback
import numpy as np
import onnx
import onnxruntime as ort
import psutil

OUT = ROOT/'artifacts/parakeet-layer-trace-dynamic-20260921'
INSPECT = ROOT/'artifacts/parakeet-trace-optimization-20260921'
JOBS = [kind+suffix for kind in ('native','managed') for suffix in ('','-repeat')]


def prepare():
    assert not OUT.exists(); spec=read(BASE/'manifest.json');verify(spec)
    original_state=read(BASE/'processes.json');inspection_state=read(INSPECT/'state.json')
    assert original_state['complete'] and original_state['code']==0 and inspection_state['complete'] and inspection_state['code']==0
    assert all(absent(i) for i in [original_state['supervisor'],inspection_state['supervisor']]+[r['worker'] for r in original_state['runs']+inspection_state['runs']])
    old=read(BASE/'analysis.json')
    assert [r['name'] for r in old['checks'] if not r['passed']]==['plain/trace-native-native-outputs','plain/trace-native-managed-outputs']
    OUT.mkdir();model=onnx.load(ROOT/spec['models']['plain'],load_external_data=False);trace=copy.deepcopy(model)
    for name in spec['outputs']['trace'][2:]:
        # Unknown axes cannot specialize the original dynamic input shapes.
        trace.graph.output.append(onnx.helper.make_tensor_value_info(name,onnx.TensorProto.FLOAT,[None,None,None]))
    stripped=copy.deepcopy(trace);del stripped.graph.output[2:]
    assert stripped.SerializeToString()==model.SerializeToString()
    target=OUT/'encoder-trace.onnx';onnx.save_model(trace,target)
    for name in {e.value for t in model.graph.initializer for e in t.external_data if e.key=='location'}:
        assert Path(name).name==name;os.link((ROOT/spec['models']['plain']).parent/name,OUT/name)
    files={rel(p):pin(p) for p in OUT.iterdir() if p.is_file()}
    files.update({rel(Path(__file__)):pin(__file__),rel(BASE/'closed.json'):pin(BASE/'closed.json')})
    files.update({rel(p):pin(p) for p in INSPECT.rglob('*') if p.is_file()})
    write(OUT/'prepared.json',dict(source=pin(__file__),jobs=JOBS,files=files,prior_manifest=pin(BASE/'manifest.json'),limits=LIMITS))
    print(json.dumps(dict(prepared=pin(OUT/'prepared.json'),jobs=JOBS)))


def verified():
    preparation=read(OUT/'prepared.json');assert preparation['jobs']==JOBS and preparation['limits']==LIMITS
    for path,expected in preparation['files'].items():assert pin(ROOT/path)==expected,path
    spec=read(BASE/'manifest.json');verify(spec);assert pin(BASE/'manifest.json')==preparation['prior_manifest']
    return spec


def worker(job):
    assert job in JOBS and psutil.Process().cpu_affinity()==[2]
    assert not any(k.lower().startswith(('lokad_','dotnet_','complus_')) for k in os.environ)
    spec=verified();assert ort.__version__==spec['onnxruntime'] and np.__version__==spec['numpy']
    folder=OUT/'outputs'/job;folder.mkdir(parents=True,exist_ok=False)
    kind=job.split('-')[0];route=spec['inputs'][kind]
    feeds=dict(audio_signal=np.load(ROOT/route['features'],allow_pickle=False),length=np.load(ROOT/route['length'],allow_pickle=False))
    before={k:v.tobytes() for k,v in feeds.items()}
    options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    for key in ('session.intra_op.allow_spinning','session.inter_op.allow_spinning'):options.add_session_config_entry(key,'0')
    if job=='native':
        options.optimized_model_filepath=str(folder/'optimized.onnx')
        options.add_session_config_entry('session.optimized_model_external_initializers_file_name','weights.bin')
        options.add_session_config_entry('session.optimized_model_external_initializers_min_size_in_bytes','1024')
    inference=ort.InferenceSession(str(OUT/'encoder-trace.onnx'),options,providers=['CPUExecutionProvider'])
    names=[v.name for v in inference.get_outputs()];assert names==spec['outputs']['trace']
    values=inference.run(None,feeds);held=[v.tobytes() for v in values]
    assert all(v.tobytes()==before[k] for k,v in feeds.items());del inference
    assert all(v.tobytes()==b for v,b in zip(values,held,strict=True))
    write(folder/'result.json',dict(job=job,complete=True,outputs=outputs(folder,names,values),inputs_unchanged=True,
                                  held_outputs_unchanged=True,onnxruntime=ort.__version__,prepared=pin(OUT/'prepared.json')))


def run():
    verified();assert not (OUT/'state.json').exists()
    parent=psutil.Process();affinity=parent.cpu_affinity();parent.cpu_affinity([0])
    state=dict(complete=False,code=None,supervisor=dict(pid=parent.pid,birth=parent.create_time()),runs=[]);save(OUT/'state.json',state)
    try:
        for job in JOBS:
            preflight=dict(available=psutil.virtual_memory().available,disk=shutil.disk_usage(OUT).free)
            assert preflight['available']>=LIMITS['preflight_available'] and preflight['disk']>=LIMITS['disk']
            row=dict(job=job,complete=False,code=None,preflight=preflight,samples=[]);state['runs'].append(row)
            child=None;identity=None;start=time.monotonic()
            try:
                env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
                with (OUT/(job+'-stdout.txt')).open('x') as stdout,(OUT/(job+'-stderr.txt')).open('x') as stderr:
                    parent.cpu_affinity([2])
                    try:
                        child=subprocess.Popen([sys.executable,'-X','utf8','-B',__file__,job],cwd=ROOT,env=env,stdout=stdout,stderr=stderr,
                            creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NO_WINDOW)
                    finally:parent.cpu_affinity([0])
                    p=psutil.Process(child.pid);identity=dict(pid=p.pid,birth=p.create_time());row['worker']=identity;save(OUT/'state.json',state)
                    while child.poll() is None:
                        try:
                            assert p.create_time()==identity['birth'] and not p.children(recursive=True)
                            sample=dict(seconds=time.monotonic()-start,rss=p.memory_info().rss,available=psutil.virtual_memory().available,
                                        disk=shutil.disk_usage(OUT).free,affinity=p.cpu_affinity())
                        except psutil.NoSuchProcess:continue
                        row['samples'].append(sample);save(OUT/'state.json',state)
                        assert sample['seconds']<LIMITS['seconds'] and sample['rss']<LIMITS['rss'] and sample['available']>=LIMITS['available'] and sample['disk']>=LIMITS['disk'] and sample['affinity']==[2]
                        time.sleep(.25)
                    row['code']=child.wait();assert row['code']==0
            except BaseException:
                if child is not None and child.poll() is None and identity is not None and not absent(identity):
                    p=psutil.Process(identity['pid'])
                    for member in p.children(recursive=True):member.kill()
                    p.kill();child.wait(timeout=10)
                raise
            finally:
                row['complete']=True
                if child is not None:row['code']=child.poll()
                save(OUT/'state.json',state)
            assert absent(identity);print('completed',job,flush=True)
        state['code']=0
    except BaseException:
        state.update(code=1,error=traceback.format_exc());raise
    finally:
        state['complete']=True;save(OUT/'state.json',state);parent.cpu_affinity(affinity)


def analyze():
    spec=verified();state=read(OUT/'state.json')
    assert state['complete'] and state['code']==0 and absent(state['supervisor'])
    assert [r['job'] for r in state['runs']]==JOBS
    for row in state['runs']:
        assert row['complete'] and row['code']==0 and absent(row['worker']) and row['samples']
        assert row['preflight']['available']>=LIMITS['preflight_available'] and row['preflight']['disk']>=LIMITS['disk']
        for s in row['samples']:
            assert s['seconds']<LIMITS['seconds'] and s['rss']<LIMITS['rss'] and s['available']>=LIMITS['available'] and s['disk']>=LIMITS['disk'] and s['affinity']==[2]
    old=read(BASE/'closed.json')
    for path,expected in old['files'].items():assert pin(ROOT/path)==expected,path
    values={};checks=[]
    for job in JOBS:
        folder=OUT/'outputs'/job;r=read(folder/'result.json')
        assert r['complete'] and r['job']==job and r['inputs_unchanged'] and r['held_outputs_unchanged'] and r['onnxruntime']==spec['onnxruntime'] and r['prepared']==pin(OUT/'prepared.json')
        assert [v['name'] for v in r['outputs']]==spec['outputs']['trace']
        values[job]={v['name']:array(folder/v['file'],v) for v in r['outputs']}
    def same(label,a,b):
        ok=a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes();checks.append(dict(name=label,passed=ok))
    for kind in ('native','managed'):
        folder=BASE/'outputs'/('native-'+kind+'-plain');r=read(folder/'result.json')
        for v in r['outputs']:same('original-'+kind+'-'+v['name'],values[kind][v['name']],array(folder/v['file'],v))
        for name in spec['outputs']['trace']:same('repeat-'+kind+'-'+name,values[kind][name],values[kind+'-repeat'][name])
    model=onnx.load(OUT/'outputs/native/optimized.onnx',load_external_data=False)
    nodes=[dict(name=n.name,op=n.op_type,domain=n.domain,inputs=list(n.input),outputs=list(n.output),
                attributes=[dict(name=a.name,sha256=hashlib.sha256(a.SerializeToString()).hexdigest()) for a in n.attribute]) for n in model.graph.node]
    original=read(INSPECT/'plain/nodes.json')['nodes']
    checks.append(dict(name='original-native-optimized-nodes',passed=nodes==original))
    qualified=all(r['passed'] for r in checks);comparisons=[]
    if qualified:
        for route,kind,native_kind in [('native-features','native','native'),('managed-features','managed','managed'),('natural-inputs','managed','native')]:
            folder=BASE/'outputs'/('managed-'+kind+'-trace');r=read(folder/'result.json')
            for v in r['outputs']:
                if v['dtype']!='float32':continue
                actual=array(folder/v['file'],v);reference=values[native_kind][v['name']]
                comparisons.append(dict(route=route,name=v['name'],**metrics(actual,reference)))
    write(OUT/'analysis.json',dict(qualified=qualified,checks=checks,comparisons=comparisons,native_nodes=len(nodes),inference_calls=4,
                                 reused_encoder_calls=8,resource_samples=sum(len(r['samples']) for r in state['runs']),peak_rss=max(s['rss'] for r in state['runs'] for s in r['samples'])))
    files={rel(p):pin(p) for p in OUT.rglob('*') if p.is_file()}
    write(OUT/'closed.json',dict(files=files,qualified=qualified,prior_closure=pin(BASE/'closed.json'),identities=[state['supervisor']]+[r['worker'] for r in state['runs']]))
    print(json.dumps(dict(qualified=qualified,failed_checks=[r['name'] for r in checks if not r['passed']],comparisons=len(comparisons),closed=pin(OUT/'closed.json'))))
    if not qualified:raise SystemExit(1)


if __name__=='__main__':
    {'prepare':prepare,'run':run,'analyze':analyze}[sys.argv[1]]() if sys.argv[1] in ('prepare','run','analyze') else worker(sys.argv[1])
