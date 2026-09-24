"""Reuse the reviewed wall observer; permit only explicit Core identities."""
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
loader=importlib.util.spec_from_file_location('phase_transport',TOOLS.parent/'managed-phase-amd/run.py')
transport=importlib.util.module_from_spec(loader);loader.loader.exec_module(transport)
BASE=ROOT/'artifacts/parakeet-slice-materialization-profile-amd-20260924'
REMOTE='/dev/shm/lokad-parakeet-slice-materialization-profile-20260924'
PRIOR=ROOT/'artifacts/parakeet-managed-phase-amd-20260924'
REMOTE_PRIOR='/dev/shm/lokad-parakeet-managed-phase-20260924/runtime-observed'
MODELS=ROOT/'artifacts/parakeet-slice-materialization-models-amd-20260924'
BUILD=ROOT/'artifacts/parakeet-slice-materialization-build-amd-v2-20260924'
APP,REMOTE_APP=transport.APP,transport.REMOTE_APP
pin,read,write,ssh=transport.pin,transport.read,transport.write,transport.ssh
PRELUDE=transport.PRELUDE.replace(transport.REMOTE,REMOTE)
transport.BASE,transport.REMOTE,transport.PRELUDE=BASE,REMOTE,PRELUDE


def prepare():
    assert not BASE.exists()
    models=read(MODELS/'closed.json');assert models['passed'] and models['analysis']==pin(MODELS/'analysis.json')
    qualification=read(MODELS/'analysis.json')
    assert read(PRIOR/'closed.json')['passed'] and read(BUILD/'build-review.json')['passed']
    current=PRIOR/'build-collected/runtime-observed';candidate=BUILD/'build-collected/source/runtime-observed/Lokad.Onnx.dll'
    assert pin(candidate)==qualification['identities']['candidate']['Lokad.Onnx.dll']
    assert pin(current/'Lokad.Onnx.dll')==qualification['identities']['selected']['Lokad.Onnx.dll']
    assert pin(current/'Lokad.Onnx.Data.dll')['sha256']=='a2a0b4901ba8e4270b57e2b176d7a656e739886275ec66d0ca036807eea16e3e'
    BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir()
    def put(name,data):
        path=bundle/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(data if isinstance(data,bytes) else data.encode())
    for path in (PRIOR/'bundle/consumer-source').iterdir():
        if not path.is_file():continue
        data=path.read_bytes()
        if path.name=='Program.cs':
            text=data.decode();before='=="672e5f303b011e27bb23097a49252c38ddee334938c75895e3c2341df0f3be35"'
            assert text.count(before)==1
            data=text.replace(before,'==Environment.GetEnvironmentVariable("PARAKEET_PHASE_CORE_SHA")')
        put('consumer-source/'+path.name,data)
    put('candidate/Lokad.Onnx.dll',candidate.read_bytes())
    bridge=(TOOLS.parent/'selected-profile-build-amd/Bridge.cs.txt').read_bytes()
    put('bridge-source/Program.cs',bridge)
    put('bridge-source/Bridge.csproj',(TOOLS.parent/'selected-profile-build-amd/Bridge.csproj').read_bytes())
    put('bridge-source/global.json',(ROOT/'global.json').read_bytes())
    put('common.py',(TOOLS.parent/'managed-phase-amd/remote.py').read_bytes())
    for name in ['remote.py','README.md']:put(name,(TOOLS/name).read_bytes())
    external={REMOTE_PRIOR+'/'+p.name:pin(p) for p in current.iterdir() if p.is_file()}
    manifest=read(APP/'collected/manifests/current-parakeet.json')
    for value in manifest['models'].values():external[value['path']]={k:value[k] for k in ['bytes','sha256']}
    for value in [manifest['reference'],*[c['pcm'] for c in manifest['cases']]]:
        external[REMOTE_APP+'/assets/'+value['path']]={k:value[k] for k in ['bytes','sha256']}
    for name in ['manifests/current-parakeet.json','runtime/protocol.py','runtime/campaign_processes.py']:
        external[REMOTE_APP+'/'+name]=pin(APP/'collected'/name)
    spec=dict(boot=1789634288.0,prior=REMOTE_PRIOR,app=REMOTE_APP,external=external,model_closure=pin(MODELS/'closed.json'),
        phase_closure=pin(PRIOR/'closed.json'),build_review=pin(BUILD/'build-review.json'),
        core=pin(current/'Lokad.Onnx.dll'),candidate_core=pin(candidate),data=pin(current/'Lokad.Onnx.Data.dll'),
        original_consumer=pin(current/'SampledAudio.dll'),minimum_pair_gain=.80,
        build_limits=dict(available_before=2*1024**3,tmpfs_before=1024**3,rss=3*1024**3,seconds=180),
        capture_limits=dict(available_before=11*1024**3,tmpfs_before=2*1024**3,rss=12*1024**3,seconds=900),
        minimum_free=1024**3,output_limit=512*1024**2,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    write(bundle/'spec.json',spec)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for path in bundle.rglob('*'):
            if path.is_file():archive.add(path,arcname=path.relative_to(bundle).as_posix(),recursive=False)
    write(BASE/'prepared.json',dict(archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),tools={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()}))
    print(json.dumps(dict(prepared=True,archive=pin(BASE/'payload.tar.gz'))))


def observe(kind):
    assert not (BASE/'closed.json').exists()
    result=ssh(PRELUDE+f'''
from remote import read,live
state=read(base/({kind!r}+'-state.json')) if (base/({kind!r}+'-state.json')).exists() else None
ids=[{read(BASE/(kind+'-deployment.json'))!r}]
if state:ids += [dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
print(json.dumps(dict(live=[i for i in ids if live(i)],complete=state and state['complete'],code=state and state['code'],
 latest=None if not state or not state['runs'] else {{k:state['runs'][-1].get(k) for k in ['name','samples','complete','code']}},error=state and state.get('error'))))
''')
    with (BASE/(kind+'-observations.jsonl')).open('a') as stream:stream.write(json.dumps(result)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    action=sys.argv[1]
    if action=='prepare':prepare()
    elif action=='stage':transport.stage()
    elif action=='observe':observe(sys.argv[2])
    else:dict(launch=transport.launch,collect=transport.collect)[action](sys.argv[2])
