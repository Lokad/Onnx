"""Freeze the one-override prototype and its complete tensor qualification."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
PRIOR_TOOLS=TOOLS.parent/'positional-copy-cost-amd'
loader=importlib.util.spec_from_file_location('copy_transport',PRIOR_TOOLS/'run.py')
original=importlib.util.module_from_spec(loader);loader.loader.exec_module(original)
pin,read,write,ssh,SSH=original.pin,original.read,original.write,original.ssh,original.SSH
BASE=ROOT/'artifacts/parakeet-slice-dense-conversion-build-amd-20260924'
REMOTE='/dev/shm/lokad-parakeet-slice-dense-conversion-build-20260924'
SOURCE=ROOT/'artifacts/parakeet-slice-dense-conversion-source-20260924'
PRODUCT=original.PRODUCT;REMOTE_PRODUCT=original.REMOTE_PRODUCT
PRELUDE=original.PRELUDE.replace(original.REMOTE,REMOTE)
transport=original.transport
transport.BASE,transport.REMOTE,transport.PRELUDE=BASE,REMOTE,PRELUDE
original.BASE,original.REMOTE,original.PRELUDE=BASE,REMOTE,PRELUDE


def source_verified():
    value=read(SOURCE/'prepared.json')
    assert value['passed'] and value['root_product_changed'] is False and not value['component_comparison_admitted']
    assert pin(SOURCE/'prepared.json')['sha256']=='a9811c6369122eb8b53ab1d88d8329928a999d172d89786a1b393c3451e91e1e'
    for name,wanted in value['source'].items():assert pin(SOURCE/'source'/name)==wanted,name
    for name,wanted in value['before'].items():assert pin(ROOT/name)==wanted,name
    return value


def prepare():
    source=source_verified();assert not BASE.exists();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir()
    def put(name,content):
        path=bundle/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:f.write(content if isinstance(content,bytes) else content.encode())
    for name in source['source']:put('source/'+name,(SOURCE/'source'/name).read_bytes())
    put('source/tests/Lokad.Onnx.Tensors.Tests/DenseCopyCandidateIdentityTests.cs',(TOOLS/'IdentityTests.cs.txt').read_bytes())
    put('bridge-source/Program.cs',(TOOLS/'Bridge.cs.txt').read_bytes())
    put('bridge-source/Bridge.csproj',(TOOLS.parent/'selected-profile-build-amd/Bridge.csproj').read_bytes())
    put('bridge-source/global.json',(ROOT/'global.json').read_bytes())
    put('common.py',(TOOLS.parent/'managed-phase-amd/remote.py').read_bytes())
    for name in ['remote.py','README.md']:put(name,(TOOLS/name).read_bytes())
    external={REMOTE_PRODUCT+'/'+p.name:pin(p) for p in PRODUCT.iterdir() if p.is_file()}
    spec=dict(boot=1789634288.0,prior=REMOTE_PRODUCT,external=external,source_prepared=pin(SOURCE/'prepared.json'),
        core=pin(PRODUCT/'Lokad.Onnx.dll'),component_comparison_admitted=False,
        build_limits=dict(available_before=2*1024**3,tmpfs_before=1024**3,rss=3*1024**3,seconds=180),
        capture_limits=dict(available_before=2*1024**3,tmpfs_before=1024**3,rss=3*1024**3,seconds=300),
        minimum_free=1024**3,output_limit=512*1024**2,expected_tests=395,expected_skipped=0,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    write(bundle/'spec.json',spec)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for path in bundle.rglob('*'):
            if path.is_file():archive.add(path,arcname=path.relative_to(bundle).as_posix(),recursive=False)
    for path in TOOLS.glob('*.py'):ast.parse(path.read_text(encoding='utf8'),str(path))
    write(BASE/'prepared.json',dict(archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),
        tools={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()},transport=pin(PRIOR_TOOLS/'run.py')))
    print(json.dumps(dict(prepared=True,archive=pin(BASE/'payload.tar.gz'),source_files=len(source['source']))))


def prepared():
    source_verified();value=read(BASE/'prepared.json')
    assert value['archive']==pin(BASE/'payload.tar.gz') and value['spec']==pin(BASE/'bundle/spec.json')
    assert value['transport']==pin(PRIOR_TOOLS/'run.py')
    for name,wanted in value['tools'].items():assert pin(TOOLS/name)==wanted,name
    for name,wanted in read(BASE/'bundle/spec.json')['files'].items():assert pin(BASE/'bundle'/name)==wanted,name


def collect(kind):
    target=BASE/(kind+'-collected');assert not target.exists()
    script=PRELUDE+f'''
from remote import read,live,pin,verify
kind={kind!r};state=read(base/(kind+'-state.json'))
assert state['complete'] and not live(state['supervisor'])
assert all(not live(dict(pid=int(p),birth=b)) for r in state['runs'] for p,b in r['members'].items())
verify()
paths=[p for p in base.rglob('*') if p.is_file() and (p.parent==base or 'logs' in p.parts or (kind=='build' and p.parent in [base/'runtime',base/'inventory'])) and p.name!='transfer.tar.gz']
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
        elif action=='observe':original.observe(sys.argv[2])
        elif action=='collect':collect(sys.argv[2])
        else:
            assert action=='launch'
            if sys.argv[2]=='capture':assert read(BASE/'build-review.json')['passed'] and read(BASE/'build-review-transferred.json')['passed']
            transport.launch(sys.argv[2])
