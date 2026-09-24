"""Freeze complete-call timing only after complete model qualification closes."""
import ast
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin, read, save

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-prepared-recurrence-timing-amd-20260924'
CALLS=ROOT/'artifacts/parakeet-prepared-recurrence-calls-amd-v2-20260924'
MODELS=ROOT/'artifacts/parakeet-prepared-recurrence-models-amd-20260924'
SOURCE=ROOT/'artifacts/parakeet-prepared-recurrence-source-20260924'
MODELS_SHA256='c0b2f61442420eed2b9afcc94263eda9917edb6de84f09ec2016d3a850077d9d'


def previous_closed():
    for folder,digest in [(CALLS,'20b3bc4b5a2d2ddb6e0834e40fb7e2bcb5220dd1f45c813a7a830466ddbc463d'),(MODELS,MODELS_SHA256)]:
        assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
    assert len(read(MODELS/'analysis.json')['results'])==8
    for role in ['selected','candidate']:
        for mode in ['512','256']:
            reports=read(MODELS/'analysis.json')['results'];native=reports[f'{role}-native-{mode}']['native'];public=reports[f'{role}-public-{mode}']
            assert native['audit_consistent'] and native['application_passed'] and native['numeric_gate_passed'] and not native['failures']
            assert (native['arrays'],native['values'])==(784,3090494) and public['passed'] and public['public_requests']==20
            if role=='candidate':assert public['complete_selected_results_exact'] and all(r['bit_identical'] for r in native['exact_selected_comparisons'])
    assert pin(SOURCE/'prepared.json')['sha256']=='b52a89c1043165de1c376b37fc5307cd003a7c8b76f0f52508c4cdefcc669eab'
    for name,wanted in read(SOURCE/'prepared.json')['before'].items():assert pin(ROOT/name)==wanted,name


def prepare():
    assert not BASE.exists();previous_closed();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    copy(ROOT/'global.json',bundle/'source/global.json')
    for name in ['Timing.cs','Timing.csproj']:copy(TOOLS/name,bundle/'source'/name)
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py','campaign_processes.py']:copy(TOOLS/name,bundle/'tools'/name)
    assert (TOOLS/'campaign_processes.py').read_bytes()==(ROOT/'tests/e5/softmax-zero-product/campaign_processes.py').read_bytes()
    copy(TOOLS/'README.md',bundle/'prospective-timing.md')
    for label,folder in [('calls',CALLS),('models',MODELS)]:
        for name in ['closed.json','payload.json','analysis.json']:copy(folder/name,bundle/'evidence'/label/name)
        copy(folder/'collected/collection.json',bundle/'evidence'/label/'collection.json')
    copy(SOURCE/'prepared.json',bundle/'evidence/source-prepared.json')
    copy(CALLS/'collected/spec.json',bundle/'spec.json')
    identities=read(bundle/'spec.json')['identities']
    for role in ['selected','candidate']:
        for name,wanted in identities[role].items():
            original=CALLS/'collected/products'/role/name;assert pin(original)==wanted
            copy(original,bundle/'products'/role/name)
    links={}
    for p in (CALLS/'collected/fixtures').iterdir():
        assert p.is_file();wanted=pin(p);originals[p.relative_to(ROOT).as_posix()]=wanted
        links['fixtures/'+p.name]=dict(source='fixtures/'+p.name,pin=wanted)
    calls=read(CALLS/'collected/fixtures/result.json')['calls']
    counts={}
    for c in calls:counts[c['name']]=counts.get(c['name'],0)+1
    assert list(counts.values())==[74,58,92,8,74,74] and len(calls)==380
    save(bundle/'stage.json',dict(passed=True,identities=identities,links=links,cases=counts,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for p in TOOLS.iterdir():
        if p.is_file():
            if p.suffix=='.py':ast.parse(p.read_text(),str(p))
            originals[p.relative_to(ROOT).as_posix()]=pin(p)
    with tarfile.open(BASE/'payload.tar.gz','w:gz',dereference=True) as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz'),linked_files=len(links),complete_call_clocks=30400)))


if __name__=='__main__':prepare()
