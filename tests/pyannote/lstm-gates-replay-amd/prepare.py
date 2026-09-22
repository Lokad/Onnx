"""Freeze exact M26 products, qualified consumer and complete original references."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin,read,save

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/pyannote-lstm-gates-replay-amd-20260923'
BUILD=ROOT/'artifacts/pyannote-lstm-gates-build-amd-20260923'
FOCUSED=ROOT/'artifacts/pyannote-lstm-gates-focused-amd-20260923'
CURRENT=ROOT/'artifacts/parakeet-current-baseline-amd-20260922'
QUALIFIED=ROOT/'artifacts/pyannote-lstm-input-blocks-amd-v2-20260922'
REPLAY=ROOT/'artifacts/pyannote-lstm-wide-replay-amd-20260923'
AMD=ROOT/'artifacts/pyannote-lstm-input-blocks-amd-20260922'
PLATFORM=ROOT/'artifacts/pyannote-lstm-platform-reference-amd-v3-20260922'
MONITOR=ROOT/'tests/parakeet/packing-budgets/common.py'
spec=importlib.util.spec_from_file_location('gates_replay_monitor',MONITOR);monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)
UPSTREAM=[(BUILD,'build','01af7cdacb5b0f0582b6b6e3739f12badfc32f02c39448715eb5804d850e33c9'),
    (FOCUSED,'focused','66dc08ab8616e9439b71a19e7303519599ebaa9f8a6142ab19702458bd7650f9'),
    (CURRENT,'current','6c65419f54f93ac43cc9ca26886dcf4bb9b6535e85f40c4db291fb9b9e1ea4bb'),
    (QUALIFIED,'qualified','bd855eedf6cf5ce8f7f738ce5d8221e47ad502e52f47d44372a4e98cee8eabe7'),
    (REPLAY,'replay','f2652e128390715d3612c0423dc363981d616154d61df62d9f661e858c72591e')]

def previous_closed():
    for folder,label,digest in UPSTREAM:
        assert pin(folder/'closed.json')['sha256']==digest;proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name

def prepare():
    assert not BASE.exists();previous_closed();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for folder,label,digest in UPSTREAM:copy(folder/'closed.json',bundle/'evidence'/(label+'-closed.json'))
    copy(ROOT/'global.json',bundle/'source/global.json')
    copy(ROOT/'tests/pyannote/lstm-wide-replay-amd/ModelReplay.cs',bundle/'evidence/qualified-ModelReplay.cs')
    copy(TOOLS/'README.md',bundle/'prospective-plan.md')
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py']:copy(TOOLS/name,bundle/'tools'/name)
    for p in TOOLS.glob('*.py'):ast.parse(p.read_text(),str(p))
    consumer=read(REPLAY/'collected/built.json')['consumer']
    assert consumer['sha256']=='1a057180ce16082fe5bea2fc50ab4f490b1e8cf75a4c743b715dc491276b4c09'
    products=dict(selected={n:pin(CURRENT/'collected/runtimes/current'/n) for n in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']},candidate=read(BUILD/'analysis.json')['built'])
    save(bundle/'stage.json',dict(passed=True,products=products,consumer=consumer,files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for p in [*TOOLS.iterdir(),MONITOR]:
        if p.is_file():originals[p.relative_to(ROOT).as_posix()]=pin(p)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'),products=products,consumer=consumer)))

if __name__=='__main__':prepare()
