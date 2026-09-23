"""Freeze a new finite-extreme driver with the exact qualified GraphRaw caller."""
import ast,importlib.util,json,shutil,tarfile
from pathlib import Path
from protocol import pin,read,save
ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/pyannote-convolution-pointer-extremes-amd-20260923'
BUILD=ROOT/'artifacts/pyannote-convolution-pointer-build-amd-20260923'
NUM=ROOT/'artifacts/pyannote-convolution-pointer-numerics-amd-20260923'
DRIVER=ROOT/'artifacts/pyannote-convolution-channel-extremes-amd-v2-20260923'
CURRENT=ROOT/'artifacts/parakeet-current-baseline-amd-20260922'
MONITOR=ROOT/'tests/parakeet/packing-budgets/common.py'
spec=importlib.util.spec_from_file_location('extreme_monitor',MONITOR);monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)
UPSTREAM=[(DRIVER,'driver','a2c99e95bdca2919cf3c370c88d55d446398d3ab37462a23433835f6cc4ea8f9'),(BUILD,'build','edff508abee43ccd09c098f0b72325482c797bf3d2a772365706c959f3ceccc9'),
    (NUM,'numerical','416a4e35c2c99af84caadb15731f06fd51b75f6dc2f84f00d2bf6592693f8a31'),
    (CURRENT,'current','6c65419f54f93ac43cc9ca26886dcf4bb9b6535e85f40c4db291fb9b9e1ea4bb')]

def previous_closed():
    for folder,label,digest in UPSTREAM:
        assert pin(folder/'closed.json')['sha256']==digest;proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name

def prepare():
    assert not BASE.exists();previous_closed();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for suffix in ['dll','deps.json','runtimeconfig.json']:
        copy(DRIVER/'collected/runtime/selected'/('ConvExtremes.'+suffix),bundle/'driver'/('ConvExtremes.'+suffix))
    copy(TOOLS/'README.md',bundle/'prospective-plan.md')
    for folder,label,digest in UPSTREAM:copy(folder/'closed.json',bundle/'evidence'/(label+'-closed.json'))
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py']:copy(TOOLS/name,bundle/'tools'/name)
    for p in TOOLS.glob('*.py'):ast.parse(p.read_text(),str(p))
    products=dict(selected={name:pin(CURRENT/'collected/runtimes/current'/name) for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']},candidate=read(BUILD/'analysis.json')['built'])
    save(bundle/'stage.json',dict(passed=True,products=products,consumer=pin(bundle/'driver/ConvExtremes.dll'),probe=read(NUM/'analysis.json')['consumers']['raw'],files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for p in [*TOOLS.iterdir(),MONITOR]:
        if p.is_file():originals[p.relative_to(ROOT).as_posix()]=pin(p)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'))))

if __name__=='__main__':prepare()
