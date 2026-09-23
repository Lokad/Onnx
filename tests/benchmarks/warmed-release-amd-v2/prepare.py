"""Freeze fixed warmed graph comparisons, preserving the inconclusive predecessor."""
import ast,difflib,json,shutil,tarfile
from pathlib import Path
from protocol import pin,read,save
ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/warmed-release-amd-v2-20260923'
OLD=ROOT/'artifacts/release-graph-baseline-amd-v2-20260923'
BUILD=ROOT/'artifacts/parakeet-first-use-kernels-build-amd-20260923'
APP=ROOT/'artifacts/parakeet-first-use-kernels-app-amd-20260923'
SHARED=ROOT/'artifacts/parakeet-first-use-kernels-shared-amd-20260923'
PYANNOTE=ROOT/'artifacts/parakeet-first-use-kernels-pyannote-amd-20260923'
GRAPH=ROOT/'artifacts/parakeet-first-use-kernels-graphs-amd-20260923'
DIAGNOSTIC=ROOT/'artifacts/graph-startup-diagnostic-amd-20260923'
FAILED=ROOT/'artifacts/warmed-release-amd-20260923'
PRIOR=dict(baseline=OLD,build=BUILD,app=APP,shared=SHARED,pyannote=PYANNOTE,graph=GRAPH,diagnostic=DIAGNOSTIC)

def previous_closed():
    failure=read(FAILED/'setup-failure.json');assert not failure['passed'] and failure['terminal'] and not failure['inference_executed'] and not failure['build_executed']
    assert pin(FAILED/'collected/collection.json')==failure['collection']
    pins={OLD:'7bfcfa23a9dcc3aa2b6bff22acf612fd140a82bff62a5d7bf891d92da420be9a',
        BUILD:'2fb4e3e587ab463a965d7cd4290ffe3f37674182bb47b7ee529d040825c3f243',
        GRAPH:'0ba2059367fdced94d678c98fd6d5fc2ae3d4d5dafd2c8dc537c442c80756a2d',
        DIAGNOSTIC:'c2f4961396ec577711ef3ac48e629e901414a21a327594921b59fb015bda9a1f'}
    for folder,wanted in pins.items():assert pin(folder/'closed.json')['sha256']==wanted
    assert read(OLD/'closed.json')['all_controls_passed'] and read(APP/'closed.json')['admitted']
    assert not read(GRAPH/'closed.json')['admitted'] and read(DIAGNOSTIC/'closed.json')['diagnostic_only']
    for folder in PRIOR.values():
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
    selected=read(ROOT/'artifacts/pyannote-winograd-product-root-amd-20260923/bundle/stage.json')
    files={n.removeprefix('source/'):v for n,v in selected['files'].items() if n.startswith('source/')};assert len(files)==420
    for name,wanted in files.items():assert pin(ROOT/name)==wanted,name

def prepare():
    previous_closed();assert not BASE.exists();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py','native.py']:copy(TOOLS/name,bundle/'tools'/name)
    original_tools=ROOT/'tests/benchmarks/release-amd-v2'
    before=(original_tools/'Program.cs').read_text();after=(TOOLS/'Program.cs').read_text()
    assert before.count('3 : 120')==before.count('index < 60')==1
    assert after==before.replace('3 : 120','3 : 780').replace('index < 60','index < 600')
    native_before=(original_tools/'native.py').read_text();native_after=(TOOLS/'native.py').read_text()
    assert native_before.count('else 120')==native_before.count('index<60')==1
    assert native_after==native_before.replace('else 120','else 780').replace('index<60','index<600')
    patch=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='original/Program.cs',tofile='warmed/Program.cs'))
    patch+=''.join(difflib.unified_diff(native_before.splitlines(True),native_after.splitlines(True),fromfile='original/native.py',tofile='warmed/native.py'))
    (bundle/'consumer.patch').write_text(patch)
    copy(TOOLS/'Program.cs',bundle/'source/consumer/Program.cs')
    for name in ['NpySupport.cs','ReleaseBenchmark.csproj']:copy(original_tools/name,bundle/'source/consumer'/name)
    copy(TOOLS/'Bridge.cs.txt',bundle/'source/bridge/Program.cs')
    copy(ROOT/'tests/parakeet/first-use-kernels-build-amd/Bridge.csproj',bundle/'source/bridge/Bridge.csproj')
    copy(ROOT/'global.json',bundle/'source/global.json')
    for label,folder in PRIOR.items():
        for name in ['closed.json','analysis.json','payload.json']:copy(folder/name,bundle/'evidence'/label/name)
        copy(folder/'collected/collection.json',bundle/'evidence'/label/'collection.json')
    copy(OLD/'bundle/cases.json',bundle/'cases.json')
    shutil.copy2(ROOT/'.agent/m45-warmed-graph-qualification-20260923.md',bundle/'prospective-plan.md')
    copy(TOOLS/'README.md',bundle/'README.md')
    copy(FAILED/'setup-failure.json',bundle/'evidence/setup-failure.json')
    old=read(OLD/'payload.json');build=read(BUILD/'analysis.json');build_payload=read(BUILD/'payload.json')
    products=dict(current={'Lokad.Onnx.dll':old['product']['Lokad.Onnx.dll']},candidate={'Lokad.Onnx.dll':build['built']['Lokad.Onnx.dll']})
    for folder in [SHARED,PYANNOTE]:
        identities=read(folder/'analysis.json')['identities']
        assert identities['selected']['Lokad.Onnx.dll']==products['current']['Lokad.Onnx.dll']
        assert identities['candidate']['Lokad.Onnx.dll']==products['candidate']['Lokad.Onnx.dll']
    external=dict(old['external'])
    for name,wanted in build_payload['external'].items():
        assert external.setdefault(name,wanted)==wanted,name
    stage=dict(passed=True,products=products,previous_consumer=read(OLD/'collected/built.json')['consumer'],
        feed=build_payload['feed'],external=external,interpreter=old['interpreter'],python_paths=old['python_paths'],
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
