"""Freeze a three-product graph comparison using the unchanged release consumer."""
import ast,json,shutil,tarfile
from pathlib import Path
from protocol import pin,read,save
ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-first-use-kernels-graphs-amd-20260923'
OLD=ROOT/'artifacts/release-graph-baseline-amd-v2-20260923'
BUILD=ROOT/'artifacts/parakeet-first-use-kernels-build-amd-20260923'
APP=ROOT/'artifacts/parakeet-first-use-kernels-app-amd-20260923'
SHARED=ROOT/'artifacts/parakeet-first-use-kernels-shared-amd-20260923'
PYANNOTE=ROOT/'artifacts/parakeet-first-use-kernels-pyannote-amd-20260923'
PRIOR=dict(baseline=OLD,build=BUILD,app=APP,shared=SHARED,pyannote=PYANNOTE)

def previous_closed():
    assert pin(OLD/'closed.json')['sha256']=='7bfcfa23a9dcc3aa2b6bff22acf612fd140a82bff62a5d7bf891d92da420be9a'
    assert read(OLD/'closed.json')['all_controls_passed']
    assert pin(BUILD/'closed.json')['sha256']=='2fb4e3e587ab463a965d7cd4290ffe3f37674182bb47b7ee529d040825c3f243'
    assert read(APP/'closed.json')['admitted']
    for folder in PRIOR.values():
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name

def prepare():
    previous_closed();assert not BASE.exists();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py','native.py']:copy(TOOLS/name,bundle/'tools'/name)
    assert (TOOLS/'native.py').read_bytes()==(ROOT/'tests/benchmarks/release-amd-v2/native.py').read_bytes()
    for label,folder in PRIOR.items():
        for name in ['closed.json','analysis.json','payload.json']:copy(folder/name,bundle/'evidence'/label/name)
        copy(folder/'collected/collection.json',bundle/'evidence'/label/'collection.json')
    copy(OLD/'bundle/cases.json',bundle/'cases.json')
    copy(TOOLS/'README.md',bundle/'prospective-plan.md')
    old=read(OLD/'payload.json');build=read(BUILD/'analysis.json')
    products=dict(current={'Lokad.Onnx.dll':old['product']['Lokad.Onnx.dll']},candidate={'Lokad.Onnx.dll':build['built']['Lokad.Onnx.dll']})
    for label,folder in [('shared',SHARED),('pyannote',PYANNOTE)]:
        identities=read(folder/'analysis.json')['identities']
        assert identities['selected']['Lokad.Onnx.dll']==products['current']['Lokad.Onnx.dll']
        assert identities['candidate']['Lokad.Onnx.dll']==products['candidate']['Lokad.Onnx.dll']
    stage=dict(passed=True,products=products,consumer=read(OLD/'collected/built.json')['consumer'],
        external=old['external'],interpreter=old['interpreter'],python_paths=old['python_paths'],
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    save(bundle/'stage.json',stage)
    for p in TOOLS.iterdir():
        if p.is_file():
            if p.suffix=='.py':ast.parse(p.read_text(),str(p))
            originals[p.relative_to(ROOT).as_posix()]=pin(p)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'))))

if __name__=='__main__':prepare()
