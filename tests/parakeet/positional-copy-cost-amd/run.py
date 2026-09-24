"""Single-use deployment of an unchanged-Core positional-copy diagnostic."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
PHASE_TOOLS=TOOLS.parent/'managed-phase-amd'
loader=importlib.util.spec_from_file_location('phase_transport',PHASE_TOOLS/'run.py')
transport=importlib.util.module_from_spec(loader);loader.loader.exec_module(transport)
pin,read,write,ssh,SSH=transport.pin,transport.read,transport.write,transport.ssh,transport.SSH
BASE=ROOT/'artifacts/parakeet-positional-copy-cost-amd-20260924'
REMOTE='/dev/shm/lokad-parakeet-positional-copy-cost-20260924'
PRIOR=ROOT/'artifacts/parakeet-projection-route-amd-20260924'
CLOSURE=ROOT/'artifacts/parakeet-projection-route-resume-amd-20260924/closed.json'
APP=ROOT/'artifacts/parakeet-observed-dense-where-app-amd-20260924'
PRODUCT=PRIOR/'build-collected/runtime-observed'
REMOTE_PRODUCT='/dev/shm/lokad-parakeet-projection-route-20260924/runtime-observed'
REMOTE_APP='/dev/shm/lokad-parakeet-observed-dense-where-app-20260924'
PRELUDE=transport.PRELUDE.replace(transport.REMOTE,REMOTE)
transport.BASE,transport.REMOTE,transport.PRELUDE=BASE,REMOTE,PRELUDE


def prerequisites():
    assert pin(CLOSURE)['sha256']=='916d4c688b19688e837a5bd5891efb31bf34a06a925e2431bdd1883b4e7e900d'
    proof=read(CLOSURE);assert proof['passed'] and proof['analysis']==pin(CLOSURE.parent/'analysis.json')
    for name in ['observations','clocks']:assert proof[name]==pin(CLOSURE.parent/(name+'.json'))
    assert pin(PRODUCT/'Lokad.Onnx.dll')['sha256']=='f95a13c58354bf07f3b7926b72903c18b1a560a56673297cb9fe001d3541b592'
    copy=read(TOOLS.parent/'observed-dense-where-results/projection-copy-20260924.json')
    for name,wanted in copy['managed_sources'].items():assert pin(ROOT/'src/Lokad.Onnx'/name)==wanted,name
    return copy


def prepare():
    import onnx
    from onnx import numpy_helper
    assert not BASE.exists();proof=prerequisites();model=ROOT/'models/parakeet-tdt-0.6b-v3/encoder-model.onnx'
    assert pin(model)==proof['source_model'] and sys.byteorder=='little'
    parsed=onnx.load(model,load_external_data=False)
    node,=[n for n in parsed.graph.node if n.name=='Constant_1723']
    tensor,=[a.t for a in node.attribute if a.HasField('t')]
    array=numpy_helper.to_array(tensor);assert list(array.shape)==[1,9999,1024] and str(array.dtype)=='float32'
    constant=array.tobytes(order='C');assert len(constant)==40955904
    original=read(APP/'collected/manifests/candidate-parakeet.json');cases=[]
    for c in original['cases']:
        frames=c['expected']['encoded_frames'];start=(5000-frames)*1024*4;count=(2*frames-1)*1024*4
        selected=constant[start:start+count];assert len(selected)==count
        cases.append(dict(name=c['name'],frames=frames,bytes=count,output_sha256=hashlib.sha256(selected).hexdigest()))
    assert len(cases)==20 and len({c['frames'] for c in cases})==19
    assert 24*sum(c['bytes'] for c in cases)==524353536
    BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir()
    def put(name,content):
        path=bundle/name;path.parent.mkdir(exist_ok=True,parents=True)
        with path.open('xb') as stream:stream.write(content if isinstance(content,bytes) else content.encode())
    put('constant.bin',constant)
    write(bundle/'cases.json',dict(cases=cases,core_sha256=pin(PRODUCT/'Lokad.Onnx.dll')['sha256'],
        constant_sha256=pin(bundle/'constant.bin')['sha256'],original_manifest=pin(APP/'collected/manifests/candidate-parakeet.json')))
    put('source/Program.cs',(TOOLS/'Program.cs.txt').read_bytes())
    refs=''.join(f'<Reference Include="{n}"><HintPath>{REMOTE_PRODUCT}/{n}.dll</HintPath></Reference>' for n in ['Lokad.Onnx','Google.Protobuf'])
    put('source/CopyCost.csproj','<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><ImplicitUsings>enable</ImplicitUsings><Nullable>enable</Nullable><TreatWarningsAsErrors>true</TreatWarningsAsErrors></PropertyGroup><ItemGroup>'+refs+'</ItemGroup></Project>')
    put('source/global.json',(ROOT/'global.json').read_bytes())
    put('common.py',(PHASE_TOOLS/'remote.py').read_bytes())
    for name in ['remote.py','README.md']:put(name,(TOOLS/name).read_bytes())
    external={REMOTE_PRODUCT+'/'+n:pin(PRODUCT/n) for n in ['Lokad.Onnx.dll','Google.Protobuf.dll']}
    accounting=REMOTE_APP+'/runtime/campaign_processes.py';external[accounting]=pin(APP/'collected/runtime/campaign_processes.py')
    limits=dict(available_before=2*1024**3,tmpfs_before=1024**3,rss=1024**3,seconds=180)
    spec=dict(boot=1789634288.0,core=pin(PRODUCT/'Lokad.Onnx.dll'),original_model=pin(model),
        previous_closure=pin(CLOSURE),copy_proof=pin(TOOLS.parent/'observed-dense-where-results/projection-copy-20260924.json'),
        accounting=accounting,external=external,feed='/dev/shm/lokad-pyannote-blocked-spatial-app-20260922/nuget-feed',
        build_limits=limits,capture_limits=limits,minimum_free=1024**3,output_limit=128*1024**2,
        order=['generic','helper','helper','generic'],corpus_ratio_limit=1.10,case_ratio_limit=1.20,
        required_component_seconds=64.70570264783333*.03,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    write(bundle/'spec.json',spec)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in bundle.rglob('*'):
            if p.is_file():archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    for p in TOOLS.glob('*.py'):ast.parse(p.read_text(encoding='utf8'),str(p))
    write(BASE/'prepared.json',dict(archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),
        tools={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()},transport=pin(PHASE_TOOLS/'run.py')))
    print(json.dumps(dict(prepared=True,archive=pin(BASE/'payload.tar.gz'),constant=pin(bundle/'constant.bin'),cases=len(cases))))


def prepared():
    prerequisites();value=read(BASE/'prepared.json')
    assert value['archive']==pin(BASE/'payload.tar.gz') and value['spec']==pin(BASE/'bundle/spec.json')
    assert value['transport']==pin(PHASE_TOOLS/'run.py')
    for name,wanted in value['tools'].items():assert pin(TOOLS/name)==wanted,name
    for name,wanted in read(BASE/'bundle/spec.json')['files'].items():assert pin(BASE/'bundle'/name)==wanted,name


def observe(kind):
    result=ssh(PRELUDE+f'''
from remote import read,live
state=read(base/({kind!r}+'-state.json')) if (base/({kind!r}+'-state.json')).exists() else None
ids=[{read(BASE/(kind+'-deployment.json'))!r}]
if state:ids += [dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
print(json.dumps(dict(live=[i for i in ids if live(i)],complete=state and state['complete'],code=state and state['code'],
 latest=None if not state or not state['runs'] else {{k:state['runs'][-1].get(k) for k in ['name','samples','complete','code']}},
 error=state and state.get('error'),stderr=(base/({kind!r}+'-supervisor.stderr')).read_text()[-4000:])))
''')
    with (BASE/(kind+'-observations.jsonl')).open('a') as stream:stream.write(json.dumps(result)+'\n')
    print(json.dumps(result))


def collect(kind):
    target=BASE/(kind+'-collected');assert not target.exists()
    script=PRELUDE+f'''
from remote import read,live,pin,verify
kind={kind!r};state=read(base/(kind+'-state.json'))
assert state['complete'] and not live(state['supervisor'])
assert all(not live(dict(pid=int(p),birth=b)) for r in state['runs'] for p,b in r['members'].items())
verify()
paths=[p for p in base.rglob('*') if p.is_file() and (p.parent==base or 'logs' in p.parts or (kind=='build' and p.parent==base/'runtime') or (kind=='capture' and 'results' in p.parts)) and p.name not in ['constant.bin','transfer.tar.gz']]
files={{p.relative_to(base).as_posix():pin(p) for p in paths}}
(base/(kind+'-collection.json')).write_text(json.dumps(dict(files=files,state=pin(base/(kind+'-state.json')),terminal=True,code=state['code'])))
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for name in [*files,kind+'-collection.json']:tar.add(base/name,arcname=name,recursive=False)
'''
    archive=BASE/(kind+'-results.tar.gz')
    with archive.open('xb') as out,(BASE/(kind+'-collection.stderr')).open('x') as err:
        result=subprocess.run(SSH+['python3','-B','-'],input=script.encode(),stdout=out,stderr=err,timeout=180,creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode==0;target.mkdir()
    with tarfile.open(archive) as tar:tar.extractall(target,filter='data')
    receipt=read(target/(kind+'-collection.json'))
    for name,wanted in receipt['files'].items():assert pin(target/name)==wanted,name
    write(BASE/(kind+'-transfer.json'),dict(passed=True,archive=pin(archive),collection=pin(target/(kind+'-collection.json'))))
    print(json.dumps(dict(code=receipt['code'],files=len(receipt['files']),archive=pin(archive))))


if __name__=='__main__':
    action=sys.argv[1]
    if action=='prepare':prepare()
    else:
        prepared()
        if action=='stage':transport.stage()
        elif action=='observe':observe(sys.argv[2])
        elif action=='collect':collect(sys.argv[2])
        else:
            assert action=='launch'
            if sys.argv[2]=='capture':assert read(BASE/'build-review.json')['passed'] and read(BASE/'build-review-transferred.json')['passed']
            transport.launch(sys.argv[2])
