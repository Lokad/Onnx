"""Stage a bounded observation of the unchanged, qualified Parakeet application."""
import ast
import base64
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-sigmoid-residual-diagnostic-amd-20260928'
REMOTE='/dev/shm/lokad-parakeet-sigmoid-residual-diagnostic-20260928'
PROFILE=ROOT/'artifacts/parakeet-transpose-axis-profile-amd-20260928'
REMOTE_PROFILE='/dev/shm/lokad-transpose-axis-profile-20260928'
EVENTS=ROOT/'artifacts/e5-direct-tier-diagnostic-v2-amd-20260925'
REMOTE_EVENTS='/dev/shm/lokad-e5-direct-tier-diagnostic-v2-20260925'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


native=load('qualified_native_adapter',TOOLS.parent/'transpose-axis-ort-profile-recovery-amd/run.py')
transport=native.transport
pin,read,write=transport.pin,transport.read,transport.write
APP,REMOTE_APP=native.APP,native.REMOTE_APP
PRELUDE=transport.PRELUDE+f'\nbase=Path({REMOTE!r})\n'
FLAGS={'DOTNET_JitDisasm':'Lokad.Onnx.CPUExecutionProvider:Sigmoid Lokad.Onnx.CPUExecutionProvider:SigmoidRationalVector',
       'DOTNET_JitDisasmWithCodeBytes':'1'}
PROVIDERS=('Microsoft-Windows-DotNETRuntime:0x1019:5,'
           'Microsoft-DotNETCore-SampleProfiler:0x0:5,'
           'Lokad-Pyannote-Diagnostic:0xffffffffffffffff:4')


def prepared():
    value=read(BASE/'prepared.json');assert value['passed']
    for name,wanted in value['inputs'].items():assert pin(ROOT/name)==wanted,name
    for name,wanted in read(BASE/'spec.json')['files'].items():assert pin(BASE/name)==wanted,name
    assert pin(BASE/'spec.json')==value['spec']
    return value


def prepare():
    assert not BASE.exists()
    context=native.qualification()
    proof=read(PROFILE/'closed.json')
    assert proof['passed'] and pin(PROFILE/'closed.json')['sha256']=='c5b1ae18221d3c583d9158dfada21f3d920434cea664d61bd99de6677da144de'
    for name,wanted in proof['files'].items():assert pin(PROFILE/name)==wanted,name
    assert pin(EVENTS/'closed.json')['sha256']=='e61941d9308e33b688f222665a0f79dd985f87b85d56123554a47202d8c531bc'
    assert read(EVENTS/'closed.json')['passed']
    inputs={};external={}
    def bind(path):inputs[path.relative_to(ROOT).as_posix()]=pin(path)
    for path in [PROFILE/'closed.json',PROFILE/'bundle/spec.json',EVENTS/'closed.json',EVENTS/'payload.json',EVENTS/'collected/collection.json',APP/'closed.json']:
        bind(path)
    prior=read(PROFILE/'bundle/spec.json')
    external.update(prior['external'])
    runtime={}
    for path in (PROFILE/'bundle/runtime-control').iterdir():
        if path.is_file():
            name=REMOTE_PROFILE+'/runtime-control/'+path.name
            external[name]=runtime[path.name]=pin(path);bind(path)
    assert runtime['Lokad.Onnx.dll']==context['built']['Lokad.Onnx.dll']
    assert runtime['Lokad.Onnx.Data.dll']==context['built']['Lokad.Onnx.Data.dll']
    assert runtime['SampledAudio.dll']['sha256']=='38ab5c7e65aa00ace50e8704348830286047e0c96ebc0a04b66c6e54d063899c'
    tool_payload=read(EVENTS/'payload.json')
    tool_files=read(EVENTS/'collected/collection.json')['files']
    for mapping,prefix in [(tool_payload['files'],'tracer/'),(tool_files,'export-runtime/')]:
        for name,wanted in mapping.items():
            if name.startswith(prefix):external[REMOTE_EVENTS+'/'+name]=wanted
    external[REMOTE_PROFILE+'/remote.py']=pin(PROFILE/'bundle/remote.py')
    for name in ['runtime/protocol.py','runtime/campaign_processes.py']:
        external[REMOTE_APP+'/'+name]=pin(APP/'collected'/name)
    source=ROOT/'artifacts/parakeet-current-profile-amd-20260923/payload/tools/pair.py.txt'
    bind(source);pair=source.read_text(encoding='utf8')
    replacements=[('env=clean_env()', 'env=clean_env(role)'),
        ("and not ready['flags']", "and ready['flags']==spec['diagnostic_flags']"),
        ("'--profile', 'dotnet-common,dotnet-sampled-thread-time',\n                            '--providers', 'Lokad-Pyannote-Diagnostic:0xffffffffffffffff:4'", "'--providers', spec['providers']"),
        ("BASE / 'tracer/dotnet-trace.dll'", "Path(spec['tracer'])"),
        ('output_bytes <= 1024**3', 'output_bytes <= 512*1024**2')]
    for before,after in replacements:
        assert pair.count(before)==1,before;pair=pair.replace(before,after)
    ast.parse(pair)
    BASE.mkdir()
    (BASE/'pair.py.txt').write_text(pair,encoding='utf8')
    for path in TOOLS.iterdir():
        if path.is_file():
            bind(path)
            if path.suffix=='.py':ast.parse(path.read_text(encoding='utf8'))
            if path.name in ['remote.py','README.md']:(BASE/path.name).write_bytes(path.read_bytes())
    value=dict(boot=1789634288.0,app=REMOTE_APP,runtime=REMOTE_PROFILE+'/runtime-control',
        common=REMOTE_PROFILE+'/remote.py',tracer=REMOTE_EVENTS+'/tracer/dotnet-trace.dll',
        exporter=REMOTE_EVENTS+'/export-runtime/DispatchEventsExport.dll',
        diagnostic_flags=FLAGS,providers=PROVIDERS,product=context['built'],runtime_files=runtime,
        profile=pin(PROFILE/'closed.json'),application=pin(APP/'closed.json'),root=context['root'],
        jobs=['control','sampled'],external=external,
        limits=dict(available_before=11*1024**3,tmpfs_before=2*1024**3,rss=12*1024**3,
                    minimum_free=1024**3,output=512*1024**2,seconds=900),
        files={p.name:pin(p) for p in BASE.iterdir() if p.is_file()},
        observer_only=True,product_rebuilt=False,consumer_rebuilt=False,pair_changes=replacements)
    write(BASE/'spec.json',value)
    write(BASE/'prepared.json',dict(passed=True,spec=pin(BASE/'spec.json'),inputs=inputs,qualification=context))
    print(json.dumps(dict(passed=True,spec=pin(BASE/'spec.json'),external_files=len(external))))


def stage():
    prepared();assert not (BASE/'staged.json').exists()
    files={name:base64.b64encode((BASE/name).read_bytes()).decode() for name in [*read(BASE/'spec.json')['files'],'spec.json']}
    result=transport.ssh(PRELUDE+f'''
import base64,importlib.util
p=Path({REMOTE_PROFILE!r})/'remote.py'
m=importlib.util.spec_from_file_location('prior_monitor',p);common=importlib.util.module_from_spec(m);m.loader.exec_module(common)
common.idle();assert not base.exists()
assert psutil.boot_time()==1789634288.0 and psutil.virtual_memory().available>=11*1024**3
assert psutil.disk_usage('/dev/shm').free>=2*1024**3
base.mkdir()
for name,data in {files!r}.items():(base/name).write_bytes(base64.b64decode(data))
sys.path.insert(0,str(base))
from remote import verify,pin
verify()
print(json.dumps(dict(passed=True,spec=pin(base/'spec.json'))))
''')
    assert result['spec']==pin(BASE/'spec.json');write(BASE/'staged.json',result);print(json.dumps(result))


def launch():
    prepared();assert read(BASE/'staged.json')['passed'] and not (BASE/'deployment.json').exists()
    result=transport.ssh(PRELUDE+'''
sys.path.insert(0,str(base))
from remote import verify,idle,pin
verify();idle();assert not (base/'state.json').exists()
with (base/'supervisor.stdout').open('x') as out,(base/'supervisor.stderr').open('x') as err:
 p=subprocess.Popen([sys.executable,'-B',str(base/'remote.py')],cwd=base,stdin=subprocess.DEVNULL,
  stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
value=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time())
(base/'deployment.json').write_text(json.dumps(value))
print(json.dumps(value))
''')
    write(BASE/'deployment.json',result);print(json.dumps(result))


def observe():
    result=transport.ssh(PRELUDE+'''
sys.path.insert(0,str(base))
from remote import read,live
s=read(base/'state.json') if (base/'state.json').exists() else None
owners=[read(base/'deployment.json')]+([] if s is None else [i for r in s['runs'] for i in r.get('processes',{}).values()])
print(json.dumps(dict(live=[i for i in owners if live(i)],complete=s and s['complete'],code=s and s['code'],
 latest=None if not s or not s['runs'] else {k:s['runs'][-1].get(k) for k in ['name','complete','code','samples']},
 error=s and s.get('error'),stderr=(base/'supervisor.stderr').read_text()[-5000:])))
''')
    with (BASE/'observations.jsonl').open('a') as stream:stream.write(json.dumps(result)+'\n')
    print(json.dumps(result));return result


def collect():
    assert not (BASE/'collected').exists() and not (BASE/'results.tar.gz').exists()
    script=PRELUDE+'''
sys.path.insert(0,str(base))
from remote import read,live,pin,verify
s=read(base/'state.json');owners=[read(base/'deployment.json')]+[i for r in s['runs'] for i in r['processes'].values()]
assert s['complete'] and not any(live(i) for i in owners);verify()
files={p.relative_to(base).as_posix():pin(p) for p in base.rglob('*') if p.is_file()}
(base/'collection.json').write_text(json.dumps(dict(terminal=True,code=s['code'],identities=owners,files=files)))
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for name in [*files,'collection.json']:tar.add(base/name,arcname=name,recursive=False)
'''
    with (BASE/'results.tar.gz').open('xb') as out,(BASE/'collection.stderr').open('x') as err:
        result=subprocess.run(transport.SSH+['python3','-B','-'],input=script.encode(),stdout=out,stderr=err,timeout=300,creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode==0,'Preserve partial collection; do not repeat inference'
    target=BASE/'collected';target.mkdir()
    with tarfile.open(BASE/'results.tar.gz') as archive:
        rows=archive.getmembers();assert all(r.isfile() and not Path(r.name).is_absolute() and '..' not in Path(r.name).parts for r in rows)
        assert len(rows)==len({r.name for r in rows});archive.extractall(target,filter='data')
    receipt=read(target/'collection.json')
    for name,wanted in receipt['files'].items():assert pin(target/name)==wanted,name
    write(BASE/'transfer.json',dict(passed=True,archive=pin(BASE/'results.tar.gz'),collection=pin(target/'collection.json')))
    print(json.dumps(dict(terminal=True,code=receipt['code'],files=len(receipt['files']))))


if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ['prepare','stage','launch','observe','collect']
    globals()[sys.argv[1]]()
