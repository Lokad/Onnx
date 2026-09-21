"""Inspect the two native optimized graphs after the failed trace control.

Creates sessions and serializes their plans; never calls InferenceSession.run.
The original inference campaign and its failed instrumentation check stay closed.
"""
from common import *
import collections
import shutil
import subprocess
import time
import traceback
import urllib.request
import onnx
import onnxruntime as ort
import psutil

OUT = ROOT/'artifacts/parakeet-trace-optimization-20260921'
URL = 'https://raw.githubusercontent.com/microsoft/onnxruntime/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h'


def worker(mode):
    spec = read(BASE/'manifest.json'); verify(spec)
    assert psutil.Process().cpu_affinity() == [2] and ort.__version__ == spec['onnxruntime']
    folder = OUT/mode; folder.mkdir()
    options = ort.SessionOptions(); options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.optimized_model_filepath = str(folder/'optimized.onnx')
    for key in ('session.intra_op.allow_spinning','session.inter_op.allow_spinning'): options.add_session_config_entry(key,'0')
    options.add_session_config_entry('session.optimized_model_external_initializers_file_name','weights.bin')
    options.add_session_config_entry('session.optimized_model_external_initializers_min_size_in_bytes','1024')
    inference = ort.InferenceSession(str(ROOT/spec['models'][mode]), options, providers=['CPUExecutionProvider'])
    assert [v.name for v in inference.get_outputs()] == spec['outputs'][mode]
    del inference
    model = onnx.load(folder/'optimized.onnx', load_external_data=False)
    nodes = [dict(name=n.name, op=n.op_type, domain=n.domain, inputs=list(n.input), outputs=list(n.output),
                  attributes=[dict(name=a.name, sha256=hashlib.sha256(a.SerializeToString()).hexdigest()) for a in n.attribute]) for n in model.graph.node]
    write(folder/'nodes.json', dict(nodes=nodes, census=dict(collections.Counter(n['op'] for n in nodes)),
          files={p.name:pin(p) for p in folder.iterdir() if p.is_file()}, inference_calls=0,
          onnxruntime=ort.__version__, model=pin(ROOT/spec['models'][mode])))


def main():
    assert not OUT.exists(); spec = read(BASE/'manifest.json'); verify(spec)
    closed = read(BASE/'closed.json'); assert not closed['trace_qualified']
    assert all(absent(i) for i in closed['identities'])
    OUT.mkdir(); (OUT/'config_keys.h').write_bytes(urllib.request.urlopen(URL,timeout=30).read())
    header = (OUT/'config_keys.h').read_text()
    assert 'session.optimized_model_external_initializers_file_name' in header and 'session.optimized_model_external_initializers_min_size_in_bytes' in header
    parent = psutil.Process(); original_affinity = parent.cpu_affinity(); parent.cpu_affinity([0])
    state = dict(complete=False, code=None, source=pin(__file__), header=dict(url=URL, **pin(OUT/'config_keys.h')),
                 previous_closure=pin(BASE/'closed.json'), supervisor=dict(pid=parent.pid,birth=parent.create_time()), runs=[])
    save(OUT/'state.json',state)
    try:
        for mode in ('plain','trace'):
            assert psutil.virtual_memory().available >= LIMITS['preflight_available'] and shutil.disk_usage(OUT).free >= LIMITS['disk']
            run = dict(mode=mode, complete=False, code=None, samples=[]);state['runs'].append(run)
            child = None; identity = None; started = time.monotonic()
            try:
                env = {k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
                with (OUT/(mode+'-stdout.txt')).open('x') as stdout, (OUT/(mode+'-stderr.txt')).open('x') as stderr:
                    parent.cpu_affinity([2])
                    try:
                        child = subprocess.Popen([sys.executable,'-X','utf8','-B',__file__,mode],cwd=ROOT,env=env,stdout=stdout,stderr=stderr,
                            creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NO_WINDOW)
                    finally: parent.cpu_affinity([0])
                    p = psutil.Process(child.pid);identity=dict(pid=p.pid,birth=p.create_time());run['worker']=identity;save(OUT/'state.json',state)
                    while child.poll() is None:
                        try:
                            assert p.create_time()==identity['birth'] and not p.children(recursive=True)
                            sample=dict(seconds=time.monotonic()-started,rss=p.memory_info().rss,available=psutil.virtual_memory().available,
                                        disk=shutil.disk_usage(OUT).free,affinity=p.cpu_affinity())
                        except psutil.NoSuchProcess: continue
                        run['samples'].append(sample);save(OUT/'state.json',state)
                        assert sample['seconds']<LIMITS['seconds'] and sample['rss']<LIMITS['rss'] and sample['available']>=LIMITS['available'] and sample['disk']>=LIMITS['disk'] and sample['affinity']==[2]
                        time.sleep(.25)
                    run['code']=child.wait();assert run['code']==0
            except BaseException:
                if child is not None and child.poll() is None and identity is not None and not absent(identity):
                    process=psutil.Process(identity['pid'])
                    for member in process.children(recursive=True): member.kill()
                    process.kill();child.wait(timeout=10)
                raise
            finally:
                run['complete']=True
                if child is not None:run['code']=child.poll()
                save(OUT/'state.json',state)
            assert absent(identity)
        a=read(OUT/'plain/nodes.json');b=read(OUT/'trace/nodes.json')
        left={n['name']:n for n in a['nodes']};right={n['name']:n for n in b['nodes']}
        assert len(left)==len(a['nodes']) and len(right)==len(b['nodes'])
        changes=[dict(name=n,plain=left.get(n),trace=right.get(n)) for n in sorted(left.keys()|right.keys()) if left.get(n)!=right.get(n)]
        write(OUT/'differences.json',dict(plain=a['census'],trace=b['census'],changes=changes,plain_nodes=len(a['nodes']),trace_nodes=len(b['nodes'])))
        state['code']=0
    except BaseException:
        state.update(code=1,error=traceback.format_exc());raise
    finally:
        state['complete']=True;save(OUT/'state.json',state);parent.cpu_affinity(original_affinity)
    print(json.dumps(dict(plain_nodes=len(a['nodes']),trace_nodes=len(b['nodes']),changes=len(changes),inference_calls=0)))


if __name__=='__main__':
    worker(sys.argv[1]) if len(sys.argv)==2 else main()
