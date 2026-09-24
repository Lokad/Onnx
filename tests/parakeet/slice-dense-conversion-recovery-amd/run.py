"""Repair only generic-type reflection; reuse the original candidate binaries."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
PRIOR_TOOLS=TOOLS.parent/'slice-dense-conversion-build-amd'
loader=importlib.util.spec_from_file_location('slice_build_transport',PRIOR_TOOLS/'run.py')
prior=importlib.util.module_from_spec(loader);loader.loader.exec_module(prior)
ORIGINAL=prior.BASE;REMOTE_ORIGINAL=prior.REMOTE
BASE=ROOT/'artifacts/parakeet-slice-dense-conversion-recovery-amd-20260924'
REMOTE='/dev/shm/lokad-parakeet-slice-dense-conversion-recovery-20260924'
pin,read,write,ssh,SSH=prior.pin,prior.read,prior.write,prior.ssh,prior.SSH
PRELUDE=prior.PRELUDE.replace(REMOTE_ORIGINAL,REMOTE)
for module in [prior,prior.original,prior.transport]:module.BASE,module.REMOTE,module.PRELUDE=BASE,REMOTE,PRELUDE


def previous():
    prior.source_verified()
    value=read(ORIGINAL/'prepared.json')
    assert value['archive']==pin(ORIGINAL/'payload.tar.gz') and value['spec']==pin(ORIGINAL/'bundle/spec.json')
    for name,wanted in value['tools'].items():assert pin(PRIOR_TOOLS/name)==wanted,name
    for name,wanted in read(ORIGINAL/'bundle/spec.json')['files'].items():assert pin(ORIGINAL/'bundle'/name)==wanted,name
    folder=ORIGINAL/'build-collected';receipt=read(folder/'build-collection.json')
    assert receipt['terminal'] and receipt['code']==1
    assert read(ORIGINAL/'build-transfer.json')['collection']==pin(folder/'build-collection.json')
    for name,wanted in receipt['files'].items():assert pin(folder/name)==wanted,name
    state=read(folder/'build-state.json');assert state['complete'] and state['code']==1
    assert state['supervisor']==read(ORIGINAL/'build-deployment.json')
    assert [r['name'] for r in state['runs']]==['sdk-version','tensors-restore','tensors-build','bridge-restore','bridge-build','inventory']
    assert [r['code'] for r in state['runs']]==[0,0,0,0,0,-6]
    assert 'Effective conversion signature or override changed' in (folder/'logs/inventory.stderr').read_text()
    assert not (folder/'built.json').exists() and not (folder/'inventory/instructions.json').exists()
    return state


def prepare():
    state=previous();assert not BASE.exists();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir()
    def put(name,content):
        path=bundle/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(content if isinstance(content,bytes) else content.encode())
    before=(PRIOR_TOOLS/'Bridge.cs.txt').read_text(encoding='utf8')
    edits=[('inherited.DeclaringType?.FullName', 'inherited.DeclaringType?.GetGenericTypeDefinition().FullName'),
        ('declared.GetBaseDefinition().DeclaringType?.FullName != inherited.DeclaringType.FullName',
         'declared.GetBaseDefinition().DeclaringType?.GetGenericTypeDefinition().FullName != inherited.DeclaringType.GetGenericTypeDefinition().FullName')]
    after=before
    for old,new in edits:assert after.count(old)==1;after=after.replace(old,new)
    restored=after
    for old,new in reversed(edits):restored=restored.replace(new,old)
    assert restored==before
    put('bridge-source/Program.cs',after)
    for name in ['Bridge.csproj','global.json']:put('bridge-source/'+name,(ORIGINAL/'bundle/bridge-source'/name).read_bytes())
    put('common.py',(ORIGINAL/'bundle/common.py').read_bytes())
    for name in ['remote.py','README.md']:put(name,(TOOLS/name).read_bytes())
    previous_spec=read(ORIGINAL/'bundle/spec.json');external=dict(previous_spec['external'])
    external.update({REMOTE_ORIGINAL+'/'+n:v for n,v in previous_spec['files'].items()})
    original_files=read(ORIGINAL/'build-collected/build-collection.json')['files']
    runtime={n:v for n,v in original_files.items() if n.startswith('runtime/')}
    assert runtime and 'runtime/Lokad.Onnx.dll' in runtime
    external.update({REMOTE_ORIGINAL+'/'+n:v for n,v in runtime.items()})
    original_test_bin=REMOTE_ORIGINAL+'/source/tests/Lokad.Onnx.Tensors.Tests/bin/Release/net10.0'
    external.update({original_test_bin+'/'+Path(n).name:v for n,v in runtime.items()})
    for n in ['build-state.json','build-collection.json']:external[REMOTE_ORIGINAL+'/'+n]=pin(ORIGINAL/'build-collected'/n)
    spec=dict(boot=previous_spec['boot'],prior=previous_spec['prior'],original=REMOTE_ORIGINAL,
        original_owner=state['supervisor'],original_members=[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()],
        external=external,core=previous_spec['core'],candidate_runtime=runtime,original_test_bin=original_test_bin,
        build_limits=previous_spec['build_limits'],capture_limits=previous_spec['capture_limits'],
        minimum_free=previous_spec['minimum_free'],output_limit=previous_spec['output_limit'],
        expected_tests=previous_spec['expected_tests'],expected_skipped=0,
        reflection_correction=edits,product_rebuilt=False,original_collection=pin(ORIGINAL/'build-collected/build-collection.json'),
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    write(bundle/'spec.json',spec)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for path in bundle.rglob('*'):
            if path.is_file():archive.add(path,arcname=path.relative_to(bundle).as_posix(),recursive=False)
    for path in TOOLS.glob('*.py'):ast.parse(path.read_text(encoding='utf8'),str(path))
    write(BASE/'prepared.json',dict(archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),
        tools={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()},prior_tools=read(ORIGINAL/'prepared.json')['tools']))
    print(json.dumps(dict(prepared=True,archive=pin(BASE/'payload.tar.gz'),candidate=runtime['runtime/Lokad.Onnx.dll'])))


def prepared():
    previous();value=read(BASE/'prepared.json')
    assert value['archive']==pin(BASE/'payload.tar.gz') and value['spec']==pin(BASE/'bundle/spec.json')
    for name,wanted in value['tools'].items():assert pin(TOOLS/name)==wanted,name
    for name,wanted in read(BASE/'bundle/spec.json')['files'].items():assert pin(BASE/'bundle'/name)==wanted,name


if __name__=='__main__':
    action=sys.argv[1]
    if action=='prepare':prepare()
    else:
        prepared()
        if action=='stage':prior.transport.stage()
        elif action=='observe':prior.original.observe(sys.argv[2])
        elif action=='collect':prior.collect(sys.argv[2])
        else:
            assert action=='launch'
            if sys.argv[2]=='capture':assert read(BASE/'build-review.json')['passed'] and read(BASE/'build-review-transferred.json')['passed']
            prior.transport.launch(sys.argv[2])
