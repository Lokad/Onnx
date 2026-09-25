"""Correct graph accounting assertions against the unchanged inspected product."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
loader=importlib.util.spec_from_file_location('original_owned',TOOLS.parent/'owned-packed-weight-build/run.py')
prior=importlib.util.module_from_spec(loader);loader.loader.exec_module(prior)
ORIGINAL,REMOTE_ORIGINAL=prior.BASE,prior.REMOTE
BASE=ROOT/'artifacts/parakeet-owned-packed-weight-tests-amd-20260925'
REMOTE='/dev/shm/lokad-parakeet-owned-packed-weight-tests-20260925'
pin,read,write,ssh=prior.pin,prior.read,prior.write,prior.ssh
PRELUDE=prior.PRELUDE.replace(REMOTE_ORIGINAL,REMOTE)
for module in [prior.prior,prior.transport]:module.BASE,module.REMOTE,module.PRELUDE=BASE,REMOTE,PRELUDE


def previous():
    prior.prepared()
    assert pin(ORIGINAL/'build-review.json')['sha256']=='5f49914888aa68589e7151a03985f4bf0396f391d1c9622fdc3424af063928b6'
    assert read(ORIGINAL/'build-review.json')['passed']
    folder=ORIGINAL/'capture-collected';receipt=read(folder/'capture-collection.json')
    assert receipt['terminal'] and receipt['code']==1
    for name,wanted in receipt['files'].items():assert pin(folder/name)==wanted,name
    state=read(folder/'capture-state.json')
    assert state['complete'] and state['code']==1 and len(state['runs'])==1
    assert state['runs'][0]['name']=='contracts-512' and state['runs'][0]['code']==1
    rows=ET.parse(folder/'logs/owned-512.trx').getroot().findall('.//{*}UnitTestResult')
    failed=[r.get('testName') for r in rows if r.get('outcome')!='Passed']
    assert len(rows)==26 and len(failed)==2
    assert all('ActualShapesUseExistingArithmetic_WithOnlyRequiredRemainderCopy' in n and 'm: 167, batched: True' in n for n in failed)
    output=(folder/'logs/contracts-512.stdout').read_text()
    assert output.count('Expected: 16777216')==output.count('Actual:   0')==2
    return state


def prepare():
    state=previous();assert not BASE.exists();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir()
    def put(name,content):
        p=bundle/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:f.write(content if isinstance(content,bytes) else content.encode())
    old=read(ORIGINAL/'bundle/spec.json');changes=[]
    test='source/tests/Lokad.Onnx.Backend.Tests/OwnedPackedWeightTests.cs'
    project='source/tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj'
    for name,wanted in old['files'].items():
        if not name.startswith('source/'):continue
        content=(ORIGINAL/'bundle'/name).read_bytes()
        if name==test:
            for before,after in [('copies.TotalCopyBytes','context.LastCopyBytes'),('scratch.TotalScratchBytes','context.LastScratchBytes')]:
                assert content.count(before.encode())==1
                content=content.replace(before.encode(),after.encode());changes.append(dict(file=name,before=before,after=after))
        if name==project:
            before=b'    <ProjectReference Include="..\\..\\src\\Lokad.Onnx\\Lokad.Onnx.csproj" />'
            assert content.count(before)==1
            refs=''.join(f'<Reference Include="{n}"><HintPath>{REMOTE_ORIGINAL}/runtime/{n}.dll</HintPath></Reference>' for n in ['Lokad.Onnx','Lokad.Onnx.Data','Google.Protobuf','FastBertTokenizer','Lokad.Tokenizers','SixLabors.ImageSharp'])
            content=content.replace(before,refs.encode())
            before=b'    <ProjectReference Include="..\\..\\src\\Lokad.Onnx.Data\\Lokad.Onnx.Data.csproj" />'
            assert content.count(before)==1;content=content.replace(before,b'')
        put(name,content)
    bridge=(prior.TOOLS/'Bridge.cs.txt').read_bytes()
    before=b'new[] { "Lokad.Onnx.dll", "Lokad.Onnx.Data.dll" }'
    assert bridge.count(before)==1
    put('bridge-source/Program.cs',bridge.replace(before,b'new[] { "Lokad.Onnx.Backend.Tests.dll" }'))
    for name in ['bridge-source/Bridge.csproj','bridge-source/global.json','common.py']:put(name,(ORIGINAL/'bundle'/name).read_bytes())
    put('remote.py',(TOOLS/'vm.py').read_bytes())
    put('plan.md',(ROOT/'.agent/m76-parakeet-owned-packed-weights-20260925.md').read_bytes())
    for name in ['build-review.json','capture-collected/capture-collection.json']:put('evidence/'+Path(name).name,(ORIGINAL/name).read_bytes())
    built=read(ORIGINAL/'build-collected/built.json')
    external={REMOTE_ORIGINAL+'/runtime/'+n:v for n,v in built['runtime'].items()}
    spec=dict(boot=old['boot'],external=external,prior=REMOTE_ORIGINAL+'/runtime',before_product=built['product'],
        before_consumer=built['consumer'],product_rebuilt=False,original_review=pin(ORIGINAL/'build-review.json'),
        failed_capture=pin(ORIGINAL/'capture-collected/capture-collection.json'),failed_release_controls=old['failed_release_controls'],
        source_edits=changes,feed=old['feed'],build_limits=old['build_limits'],capture_limits=old['capture_limits'],
        minimum_free=old['minimum_free'],output_limit=old['output_limit'],expected_tests={'512':7,'256':26,'scalar':2},
        original_identities=[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()],
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    assert len(changes)==2
    write(bundle/'spec.json',spec)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in bundle.rglob('*'):
            if p.is_file():archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    for p in TOOLS.glob('*.py'):ast.parse(p.read_text(),str(p))
    write(BASE/'prepared.json',dict(archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),
        tools={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()}))
    print(json.dumps(dict(prepared=True,archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),product=built['product'])))


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
        else:
            kind=sys.argv[2];assert kind in ['build','capture']
            if action=='launch':
                if kind=='capture':assert read(BASE/'build-review-transferred.json')['passed']
                prior.transport.launch(kind)
            else:{'observe':prior.prior.observe,'collect':prior.prior.collect}[action](kind)
