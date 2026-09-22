"""Freeze a five-replacement replay adapter, exact products and qualified references."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin,read,save

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/pyannote-lstm-wide-replay-amd-20260923'
BUILD=ROOT/'artifacts/pyannote-lstm-wide-build-amd-20260922'
FOCUSED=ROOT/'artifacts/pyannote-lstm-wide-focused-amd-20260923'
CURRENT=ROOT/'artifacts/parakeet-current-baseline-amd-20260922'
QUALIFIED=ROOT/'artifacts/pyannote-lstm-input-blocks-amd-v2-20260922'
BRIDGE=ROOT/'artifacts/pyannote-kernel-loop-numerics-amd-v2-20260922'
AMD=ROOT/'artifacts/pyannote-lstm-input-blocks-amd-20260922'
PLATFORM=ROOT/'artifacts/pyannote-lstm-platform-reference-amd-v3-20260922'
MONITOR=ROOT/'tests/parakeet/packing-budgets/common.py'
spec=importlib.util.spec_from_file_location('wide_replay_monitor',MONITOR);monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)


def previous_closed():
    for folder,digest in [(BUILD,'cd9d2839d30d55f2a1d020da040163e1920dc97991e30584b4cdc72593e1d338'),
            (FOCUSED,'78463bb45456b6588d81ff5821304a46774d2d7e9cca05c20d2d2a862b4d453f'),
            (CURRENT,'6c65419f54f93ac43cc9ca26886dcf4bb9b6535e85f40c4db291fb9b9e1ea4bb'),
            (QUALIFIED,'bd855eedf6cf5ce8f7f738ce5d8221e47ad502e52f47d44372a4e98cee8eabe7'),
            (BRIDGE,'551fb5db9dd20c9e5e9c3737a4f311779226bdbc92c53f4cff835f6277799dda')]:
        assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name


def adapted_source():
    source=(ROOT/'tests/pyannote/lstm-input-blocks-v6/ModelReplay.cs').read_text()
    changes=[('256-or-512-or-scalar','256-or-512-or-scalar-or-simd'),
        ('args[4] is "256" or "512" or "scalar"','args[4] is "256" or "512" or "scalar" or "simd"'),
        ('Avx512F.IsSupported == (args[4] == "512")','Avx512F.IsSupported == (args[4] is "512" or "simd")'),
        ('var options = ExecutionOptions.Memory;','var options = args[4] == "simd" ? ExecutionOptions.Memory with { Tensor = TensorExecutionOptions.Simd } : ExecutionOptions.Memory;'),
        ('+(args[3]=="candidate"?8192:0)','+8192')]
    for before,after in changes:assert source.count(before)==1;source=source.replace(before,after)
    return source


def prepare():
    assert not BASE.exists();previous_closed();assert (TOOLS/'ModelReplay.cs').read_text()==adapted_source()
    BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);originals[source.relative_to(ROOT).as_posix()]=pin(source)
    copy(TOOLS/'ModelReplay.cs',bundle/'source/consumer/ModelReplay.cs');copy(ROOT/'global.json',bundle/'source/global.json')
    project=(ROOT/'artifacts/pyannote-lstm-input-blocks-v2-20260922/consumer/ModelReplay.csproj').read_text()
    assert project.count('../selected-runtime/')==2
    (bundle/'source/consumer/ModelReplay.csproj').write_text(project.replace('../selected-runtime/','../../runtime/selected/'),encoding='utf8')
    for p in (BRIDGE/'bundle/bridge').iterdir():
        if p.is_file():copy(p,bundle/'bridge'/p.name)
    for folder,label in [(BUILD,'build'),(FOCUSED,'focused'),(CURRENT,'current'),(QUALIFIED,'qualified')]:copy(folder/'closed.json',bundle/'evidence'/(label+'-closed.json'))
    copy(ROOT/'tests/pyannote/lstm-input-blocks-v6/ModelReplay.cs',bundle/'evidence/original-ModelReplay.cs')
    copy(TOOLS/'README.md',bundle/'prospective-plan.md')
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py']:copy(TOOLS/name,bundle/'tools'/name)
    for p in TOOLS.glob('*.py'):ast.parse(p.read_text(),str(p))
    old=read(QUALIFIED/'payload.json')
    products=dict(selected={n:pin(CURRENT/'collected/runtimes/current'/n) for n in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']},candidate=read(BUILD/'analysis.json')['built'])
    stage=dict(passed=True,products=products,previous_consumer=old['consumer'],files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    save(bundle/'stage.json',stage)
    for p in [*TOOLS.iterdir(),MONITOR,ROOT/'tests/pyannote/lstm-input-blocks-amd-v2/audit.py']:
        if p.is_file():originals[p.relative_to(ROOT).as_posix()]=pin(p)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'),products=products)))


if __name__=='__main__':prepare()
