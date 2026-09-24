"""Serialize original-output ORT graphs; retain metadata without duplicate weights."""
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REMOTE = '/dev/shm/lokad-parakeet-ort-graphs-v2-20260924'


def pin(path):
    path = Path(path)
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf8')


def worker(app, graph, folder):
    import importlib.util
    import gc
    assert os.sched_getaffinity(0) == {2}
    manifest = json.loads((app/'manifests/current-parakeet.json').read_text())
    for spec in manifest['models'].values():
        assert pin(spec['path']) == {k: spec[k] for k in ['bytes', 'sha256']}
    for spec in manifest['native_binaries'].values():
        assert pin(spec['path']) == {k: spec[k] for k in ['bytes', 'sha256']}
    adapter_path = (app/'assets'/manifest['adapter']['path']).resolve()
    assert pin(adapter_path) == {k: manifest['adapter'][k] for k in ['bytes', 'sha256']}
    source = importlib.util.spec_from_file_location('original_adapter', adapter_path)
    adapter = importlib.util.module_from_spec(source); source.loader.exec_module(adapter)
    ort = adapter.ort
    assert ort.__version__ == '1.29.0'
    original = ort.InferenceSession
    path = Path(manifest['models'][manifest['graphs'][graph]]['path'])
    folder.mkdir()

    def serialize(model_path, options, providers):
        assert Path(model_path) == path and providers == ['CPUExecutionProvider']
        assert options.intra_op_num_threads == options.inter_op_num_threads == 1
        assert options.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL
        assert options.graph_optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        assert not options.enable_profiling and not options.optimized_model_filepath
        for key in ('session.intra_op.allow_spinning', 'session.inter_op.allow_spinning'):
            assert options.get_session_config_entry(key) == '0'
        options.optimized_model_filepath = str(folder/'optimized.onnx')
        options.add_session_config_entry('session.optimized_model_external_initializers_file_name', 'weights.bin')
        options.add_session_config_entry('session.optimized_model_external_initializers_min_size_in_bytes', '1024')
        return original(model_path, options, providers=providers)

    ort.InferenceSession = serialize
    started = time.perf_counter()
    session = adapter.session(path)
    value = dict(graph=graph, source_model=pin(path), build_info=ort.get_build_info(), inference_calls=0,
        seconds=time.perf_counter()-started, optimized_model=pin(folder/'optimized.onnx'),
        inputs=[dict(name=v.name, shape=v.shape, type=v.type) for v in session.get_inputs()],
        outputs=[dict(name=v.name, shape=v.shape, type=v.type) for v in session.get_outputs()])
    del session; gc.collect()
    sidecar = (folder/'weights.bin').resolve()
    assert sidecar.parent == folder.resolve() and sidecar.is_file()
    value['retired_scratch'] = dict(file=sidecar.name, **pin(sidecar))
    save(folder/'before-scratch-retirement.json', value)
    sidecar.unlink()
    value['scratch_retired'] = not sidecar.exists()
    save(folder/'result.json', value)


def local(remaining=False):
    import ast
    import onnx
    from collections import Counter
    from run import BASE as PRIOR, APP, ROOT, SSH, SITE, pin as original_pin, read
    proof = read(PRIOR/'closed.json')
    assert proof['passed'] and proof['analysis'] == original_pin(PRIOR/'analysis.json')
    prior = read(PRIOR/'collected/collection.json')
    for name, wanted in prior['files'].items():
        assert original_pin(PRIOR/'collected'/name) == wanted
    graphs = ['decoder', 'frontend'] if remaining else ['encoder', 'decoder', 'frontend']
    remote = '/dev/shm/lokad-parakeet-ort-small-graphs-20260924' if remaining else REMOTE
    out = ROOT/('artifacts/parakeet-ort-small-graphs-amd-20260924' if remaining else 'artifacts/parakeet-ort-graphs-amd-v2-20260924')
    limits = dict(preflight_available=(2 if remaining else 12)*1024**3,
                  preflight_tmpfs=(2 if remaining else 4)*1024**3,
                  available=1024**3, tmpfs=1024**3, rss=(1 if remaining else 12)*1024**3,
                  output=256*1024**2 if remaining else 3*1024**3, seconds=180)
    assert not out.exists(); out.mkdir()
    data = Path(__file__).read_bytes(); ast.parse(data)
    (out/'graphs.py').write_bytes(data)
    source = base64.b64encode(data).decode('ascii')
    payload = read(APP/'payload.json')
    spec = dict(previous=pin(PRIOR/'closed.json'), source=pin(__file__),
        original_preflight_refusal=pin(ROOT/'artifacts/parakeet-ort-graphs-amd-20260924/closed.json'),
        remote_trace_retirement=pin(ROOT/'artifacts/parakeet-ort-remote-profile-retention-20260924/closed.json'),
        previous_owners=proof['terminal_owners'], payload=pin(APP/'payload.json'),
        graphs=graphs, no_inference=True, limits=limits)
    if remaining:
        earlier = read(ROOT/'artifacts/parakeet-ort-graphs-amd-v2-20260924/transfer.json')['state']
        assert earlier['complete'] and earlier['code'] == 1 and len(earlier['runs']) == 1
        assert earlier['runs'][0]['graph'] == 'encoder' and earlier['runs'][0]['code'] == 0
        spec['completed_encoder'] = pin(ROOT/'artifacts/parakeet-ort-graphs-amd-v2-20260924/collected/encoder/result.json')
    save(out/'prepared.json', spec)
    script = f'''import os,sys,json,time,subprocess,base64,hashlib,traceback
from pathlib import Path
sys.path.insert(0,{SITE!r})
import psutil
os.sched_setaffinity(0,{{0}})
prior=Path('/dev/shm/lokad-parakeet-ort-diagnosis-20260924')
sys.path.insert(0,str(prior))
from remote import live,pin,read,save
assert all(not live(i) for i in {proof['terminal_owners']!r})
assert read(prior/'state.json')['complete'] and read(prior/'state.json')['code']==0
app=Path('/dev/shm/lokad-parakeet-prepared-recurrence-app-20260924')
assert pin(app/'payload.json')=={pin(APP/'payload.json')!r}
base=Path({remote!r});assert not base.exists();base.mkdir()
(base/'graphs.py').write_bytes(base64.b64decode({source!r}))
assert pin(base/'graphs.py')=={pin(__file__)!r}
own=psutil.Process();state=dict(complete=False,code=None,supervisor=dict(pid=own.pid,birth=own.create_time()),runs=[])
save(base/'state.json',state)
env={{k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}}
env.pop('PYTHONOPTIMIZE',None)
env.update(PYTHONPATH=os.pathsep.join({payload['python_paths']!r}),PYTHONDONTWRITEBYTECODE='1')
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='1'
try:
 for graph in {graphs!r}:
  pre=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage(base).free)
  save(base/(graph+'-preflight.json'),pre)
  assert pre['available']>={limits['preflight_available']} and pre['tmpfs']>={limits['preflight_tmpfs']}
  row=dict(graph=graph,preflight=pre,complete=False,code=None,samples=[]);state['runs'].append(row)
  started=time.monotonic();child=None
  try:
   with (base/(graph+'.stdout')).open('x') as stdout,(base/(graph+'.stderr')).open('x') as stderr:
    os.sched_setaffinity(0,{{2}})
    try:child=subprocess.Popen([sys.executable,'-B',str(base/'graphs.py'),'worker',str(app),graph,str(base/graph)],env=env,cwd=base,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
    finally:os.sched_setaffinity(0,{{0}})
    p=psutil.Process(child.pid);row['owner']=dict(pid=p.pid,birth=p.create_time());save(base/'state.json',state)
    while child.poll() is None:
     try:
      assert p.create_time()==row['owner']['birth'] and not p.children(recursive=True)
      threads=[]
      for t in p.threads():
       try:threads.append(sorted(os.sched_getaffinity(t.id)))
       except ProcessLookupError:pass
      sample=dict(seconds=time.monotonic()-started,rss=p.memory_info().rss,available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage(base).free,threads=threads,
       output=sum(f.stat().st_size for f in base.rglob('*') if f.is_file()))
     except (psutil.NoSuchProcess,FileNotFoundError):continue
     row['samples'].append(sample);save(base/'state.json',state)
     assert sample['seconds']<180 and sample['rss']<{limits['rss']} and sample['available']>=1024**3 and sample['tmpfs']>=1024**3 and sample['output']<{limits['output']}
     assert all(a==[2] for a in threads)
     time.sleep(.5)
    row['code']=child.wait();assert row['code']==0,graph
   assert not live(row['owner'])
  except BaseException:
   if 'owner' in row and live(row['owner']):psutil.Process(row['owner']['pid']).kill()
   if child is not None:child.wait(timeout=15)
   raise
  finally:
   row.update(complete=True,seconds=time.monotonic()-started,code=None if child is None else child.poll());save(base/'state.json',state)
 state['code']=0
except BaseException:state.update(code=1,error=traceback.format_exc())
finally:state['complete']=True;save(base/'state.json',state)
files={{f.relative_to(base).as_posix():pin(f) for f in base.rglob('*') if f.is_file() and f.name!='weights.bin'}}
assert all(not live(r['owner']) for r in state['runs'] if 'owner' in r)
print(json.dumps(dict(state=state,files=files,data={{n:base64.b64encode((base/n).read_bytes()).decode('ascii') for n in files}})))
'''
    compile(script, 'original-ort-graph-serialization', 'exec')
    with (out/'response.json').open('xb') as stdout, (out/'transport.stderr').open('x') as stderr:
        p = subprocess.run(SSH+['python3', '-B', '-'], input=script.encode(), stdout=stdout, stderr=stderr,
                           timeout=660, creationflags=subprocess.CREATE_NO_WINDOW)
    assert p.returncode == 0, 'Preserve partial transfer and inspect existing owners; never relaunch blindly'
    value = read(out/'response.json'); collected = out/'collected'; collected.mkdir()
    for name, content in value.pop('data').items():
        target = (collected/name).resolve(); assert target.is_relative_to(collected.resolve())
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(base64.b64decode(content))
        assert pin(target) == value['files'][name]
    save(out/'transfer.json', value)
    assert value['state']['complete'] and value['state']['code'] == 0, value['state'].get('error')
    profile = read(PRIOR/'analysis.json'); reports = {}
    for graph in spec['graphs']:
        folder = collected/graph; result = read(folder/'result.json')
        assert result['scratch_retired'] and result['inference_calls'] == 0
        assert pin(folder/'optimized.onnx') == result['optimized_model']
        model = onnx.load(folder/'optimized.onnx', load_external_data=False)
        nodes = {node.name: dict(op=node.op_type, domain=node.domain, inputs=list(node.input), outputs=list(node.output),
            attributes=[dict(name=a.name, type=a.type, value=str(onnx.helper.get_attribute_value(a))) for a in node.attribute]) for node in model.graph.node}
        assert len(nodes) == len(model.graph.node)
        observed = profile['profiles'][graph]['nodes']
        for name, node in observed.items():
            assert name in nodes and nodes[name]['op'] == node['op'], (graph, name)
        assert set(nodes) == set(observed), 'Serialized nodes differ from executed original graph'
        setup = next(r for r in read(PRIOR/'collected/control/observation.json')['setup'] if r['graph'] == graph)
        assert result['inputs'] == setup['inputs'] and result['outputs'] == setup['outputs']
        reports[graph] = dict(result=result, nodes=nodes, census=dict(Counter(n['op'] for n in nodes.values())),
            initializers={t.name: dict(shape=list(t.dims), dtype=t.data_type, external={e.key:e.value for e in t.external_data}) for t in model.graph.initializer})
    save(out/'analysis.json', dict(passed=True, graphs=reports, source=spec, inference_calls=0))
    save(out/'closed.json', dict(passed=True, analysis=pin(out/'analysis.json'), transfer=pin(out/'transfer.json'),
                                source=pin(out/'prepared.json'), original_profile=pin(PRIOR/'closed.json')))
    print(json.dumps(dict(passed=True, closed=pin(out/'closed.json'), graphs={k:v['census'] for k,v in reports.items()})))


if __name__ == '__main__':
    if len(sys.argv) == 5 and sys.argv[1] == 'worker':
        worker(Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]))
    else:
        assert (len(sys.argv) == 1 or sys.argv[1:] == ['--remaining']) and os.name == 'nt'
        local(remaining=len(sys.argv) == 2)
