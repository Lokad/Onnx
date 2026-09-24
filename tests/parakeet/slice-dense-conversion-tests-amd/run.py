"""Correct a test oracle without rebuilding the inspected product."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
PRIOR_TOOLS=TOOLS.parent/'slice-dense-conversion-recovery-amd'
loader=importlib.util.spec_from_file_location('slice_recovery_transport',PRIOR_TOOLS/'run.py')
prior=importlib.util.module_from_spec(loader);loader.loader.exec_module(prior)
RECOVERY=prior.BASE;REMOTE_RECOVERY=prior.REMOTE;ORIGINAL=prior.ORIGINAL
BASE=ROOT/'artifacts/parakeet-slice-dense-conversion-tests-amd-20260924'
REMOTE='/dev/shm/lokad-parakeet-slice-dense-conversion-tests-20260924'
pin,read,write,ssh,SSH=prior.pin,prior.read,prior.write,prior.ssh,prior.SSH
PRELUDE=prior.PRELUDE.replace(REMOTE_RECOVERY,REMOTE)
for module in [prior.prior,prior.prior.original,prior.prior.transport]:module.BASE,module.REMOTE,module.PRELUDE=BASE,REMOTE,PRELUDE


def previous():
    prior.prepared()
    review=RECOVERY/'build-review.json'
    assert pin(review)['sha256']=='349ae125453183b0b04202dd6408c789760e1748ec863fc51c44d81208f17cbd'
    assert read(review)['passed'] and not read(review)['product_rebuilt']
    folder=RECOVERY/'capture-collected';receipt=read(folder/'capture-collection.json')
    assert receipt['terminal'] and receipt['code']==1
    for name,wanted in receipt['files'].items():assert pin(folder/name)==wanted,name
    state=read(folder/'capture-state.json');assert state['complete'] and state['code']==1
    assert state['supervisor']==read(RECOVERY/'capture-deployment.json')
    assert len(state['runs'])==1 and state['runs'][0]['name']=='tensors-512' and state['runs'][0]['code']==1
    ns={'t':'http://microsoft.com/schemas/VisualStudio/TeamTest/2010'}
    tree=ET.parse(folder/'logs/tensors-512.trx');rows=tree.findall('.//t:UnitTestResult',ns)
    assert len(rows)==395 and sum(r.attrib['outcome']=='Passed' for r in rows)==394
    failure,=[r for r in rows if r.attrib['outcome']!='Passed']
    assert failure.attrib['testName']=='Lokad.Onnx.Tensors.Tests.SliceDenseConversionTests.EmptyAndReversedStorageRemainSupported'
    assert 'Expected: [2, 3, 4, 5]' in (folder/'logs/tensors-512.stdout').read_text()
    assert 'Actual:   [2, 4, 3, 5]' in (folder/'logs/tensors-512.stdout').read_text()
    return state


def prepare():
    state=previous();assert not BASE.exists();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir()
    def put(name,content):
        path=bundle/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(content if isinstance(content,bytes) else content.encode())
    previous_spec=read(RECOVERY/'bundle/spec.json');original_spec=read(ORIGINAL/'bundle/spec.json')
    test='source/tests/Lokad.Onnx.Tensors.Tests/SliceDenseConversionTests.cs'
    project='source/tests/Lokad.Onnx.Tensors.Tests/Lokad.Onnx.Tensors.Tests.csproj'
    before='Assert.Equal(new[]{2,3,4,5},result.ToArray());Assert.True(result.IsReversedStride);'
    after='Assert.Equal(new[]{2,4,3,5},result.ToArray());\n        Assert.Equal(new[]{2,3,4,5},result.Buffer.ToArray());Assert.True(result.IsReversedStride);'
    changed=[]
    for name,wanted in original_spec['files'].items():
        if not name.startswith('source/'):continue
        content=(ORIGINAL/'bundle'/name).read_bytes();assert pin(ORIGINAL/'bundle'/name)==wanted
        if name==test:
            text=content.decode();assert text.count(before)==1;content=text.replace(before,after).encode();changed.append(name)
        if name==project:
            text=content.decode();needle='<ProjectReference Include="..\\..\\src\\Lokad.Onnx\\Lokad.Onnx.csproj" />'
            assert text.count(needle)==1
            refs=''.join(f'<Reference Include="{n}"><HintPath>{REMOTE_RECOVERY}/runtime/{n}.dll</HintPath></Reference>' for n in ['Lokad.Onnx','Google.Protobuf'])
            content=text.replace(needle,refs).encode();changed.append(name)
        put(name,content)
    assert set(changed)=={test,project}
    put('common.py',(ORIGINAL/'bundle/common.py').read_bytes())
    for name in ['remote.py','README.md']:put(name,(TOOLS/name).read_bytes())
    built=read(RECOVERY/'build-collected/built.json')
    external={REMOTE_RECOVERY+'/runtime/'+n:pin(RECOVERY/'build-collected/runtime'/n) for n in ['Lokad.Onnx.dll','Google.Protobuf.dll']}
    for name in ['build-review.json','built.json','capture-state.json','capture-collection.json']:
        location=RECOVERY/name if name=='build-review.json' else RECOVERY/('build-collected' if name=='built.json' else 'capture-collected')/name
        external[REMOTE_RECOVERY+'/'+name]=pin(location)
    spec=dict(boot=previous_spec['boot'],external=external,core=built['core'],product_rebuilt=False,
        build_limits=previous_spec['build_limits'],capture_limits=previous_spec['capture_limits'],
        minimum_free=previous_spec['minimum_free'],output_limit=previous_spec['output_limit'],expected_tests=395,expected_skipped=0,
        corrected_test=dict(file=test,before=before,after=after),source_project_binding=project,
        original_owner=state['supervisor'],original_members=[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()],
        compiled_review=pin(RECOVERY/'build-review.json'),failed_capture=pin(RECOVERY/'capture-collected/capture-collection.json'),
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    write(bundle/'spec.json',spec)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for path in bundle.rglob('*'):
            if path.is_file():archive.add(path,arcname=path.relative_to(bundle).as_posix(),recursive=False)
    for path in TOOLS.glob('*.py'):ast.parse(path.read_text(encoding='utf8'),str(path))
    write(BASE/'prepared.json',dict(archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),tools={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()}))
    print(json.dumps(dict(prepared=True,archive=pin(BASE/'payload.tar.gz'),core=built['core'])))


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
        if action=='stage':prior.prior.transport.stage()
        elif action=='observe':prior.prior.original.observe(sys.argv[2])
        elif action=='collect':prior.prior.collect(sys.argv[2])
        else:
            assert action=='launch'
            if sys.argv[2]=='capture':assert read(BASE/'build-review.json')['passed'] and read(BASE/'build-review-transferred.json')['passed']
            prior.prior.transport.launch(sys.argv[2])
