"""Run only the missing AVX512-disabled suite with the established runtime flag."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
ORIGINAL=ROOT/'artifacts/parakeet-slice-dense-conversion-tests-amd-20260924'
REMOTE_ORIGINAL='/dev/shm/lokad-parakeet-slice-dense-conversion-tests-20260924'
PRIOR_TOOLS=TOOLS.parent/'slice-dense-conversion-build-amd'
loader=importlib.util.spec_from_file_location('slice_transport',PRIOR_TOOLS/'run.py')
prior=importlib.util.module_from_spec(loader);loader.loader.exec_module(prior)
BASE=ROOT/'artifacts/parakeet-slice-dense-conversion-isa-amd-20260924'
REMOTE='/dev/shm/lokad-parakeet-slice-dense-conversion-isa-20260924'
pin,read,write,ssh,SSH=prior.pin,prior.read,prior.write,prior.ssh,prior.SSH
PRELUDE=prior.PRELUDE.replace(prior.REMOTE,REMOTE)
for module in [prior,prior.original,prior.transport]:module.BASE,module.REMOTE,module.PRELUDE=BASE,REMOTE,PRELUDE


def previous():
    prior.source_verified()
    value=read(ORIGINAL/'prepared.json')
    for name,wanted in value['tools'].items():assert pin(TOOLS.parent/'slice-dense-conversion-tests-amd'/name)==wanted,name
    assert value['archive']==pin(ORIGINAL/'payload.tar.gz') and value['spec']==pin(ORIGINAL/'bundle/spec.json')
    for name,wanted in read(ORIGINAL/'bundle/spec.json')['files'].items():assert pin(ORIGINAL/'bundle'/name)==wanted,name
    assert pin(ORIGINAL/'build-review.json')['sha256']=='0eeef7982a09859c8f791bc361d900ab04941ada148e5600527978732b5351d4'
    assert read(ORIGINAL/'build-review.json')['passed']
    folder=ORIGINAL/'capture-collected';receipt=read(folder/'capture-collection.json');state=read(folder/'capture-state.json')
    for name,wanted in receipt['files'].items():assert pin(folder/name)==wanted,name
    assert receipt['terminal'] and receipt['code']==state['code']==1 and state['complete']
    assert [r['name'] for r in state['runs']]==['tensors-512','tensors-256'] and [r['code'] for r in state['runs']]==[0,1]
    assert 'Expected: False' in (folder/'logs/tensors-256.stdout').read_text() and 'Actual:   True' in (folder/'logs/tensors-256.stdout').read_text()
    return state


def prepare():
    state=previous();assert not BASE.exists();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir()
    def put(name,content):
        with (bundle/name).open('xb') as stream:stream.write(content if isinstance(content,bytes) else content.encode())
    put('common.py',(ORIGINAL/'bundle/common.py').read_bytes())
    for name in ['remote.py','README.md']:put(name,(TOOLS/name).read_bytes())
    original_spec=read(ORIGINAL/'bundle/spec.json');built=read(ORIGINAL/'build-collected/built.json')
    external=dict(original_spec['external']);external.update({REMOTE_ORIGINAL+'/'+n:v for n,v in original_spec['files'].items()})
    for name,wanted in built['runtime_files'].items():
        external[REMOTE_ORIGINAL+'/'+name]=wanted
        external[built['test_bin']+'/'+Path(name).name]=wanted
    for name in ['built.json','build-review.json','capture-state.json','capture-collection.json']:
        path=ORIGINAL/name if name=='build-review.json' else ORIGINAL/('build-collected' if name=='built.json' else 'capture-collected')/name
        external[REMOTE_ORIGINAL+'/'+name]=pin(path)
    spec=dict(boot=original_spec['boot'],original=REMOTE_ORIGINAL,external=external,core=built['core'],consumer=built['consumer'],
        original_owner=state['supervisor'],original_members=[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()],
        capture_limits=original_spec['capture_limits'],minimum_free=original_spec['minimum_free'],output_limit=original_spec['output_limit'],
        expected_tests=395,expected_skipped=0,instruction_environment={'DOTNET_EnableAVX512':'0'},
        failed_capture=pin(ORIGINAL/'capture-collected/capture-collection.json'),build_review=pin(ORIGINAL/'build-review.json'),
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.iterdir() if p.is_file()})
    write(bundle/'spec.json',spec)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for path in bundle.iterdir():archive.add(path,arcname=path.name,recursive=False)
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
        if action=='stage':prior.transport.stage()
        elif action=='launch':prior.transport.launch('capture')
        elif action=='observe':prior.original.observe('capture')
        elif action=='collect':prior.collect('capture')
        else:raise ValueError(action)
