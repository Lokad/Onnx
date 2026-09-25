"""Freeze two original long-e5 processes and the qualified external counter."""
import ast
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin, read, save, LIMITS

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/e5-frequency-diagnostic-amd-20260925'
OLD = ROOT/'artifacts/e5-repeatability-diagnostic-amd-20260925'
PROOF = ROOT/'artifacts/e5-frequency-proof-v3-amd-20260925'
REMOTE_OLD = '/dev/shm/lokad-e5-repeatability-diagnostic-20260925'


def previous_closed():
    assert pin(OLD/'closed.json')['sha256']=='fe8ec805d18864a3a8d8b5a052543a3eef466535adc2769efc8c6c6747338a7b'
    assert pin(PROOF/'closed.json')['sha256']=='8f63ad0396ef8fdb762afa2cf88bece14f0d6e9b49c4b033a1ad5906a323778f'
    for folder in [OLD, PROOF]:
        closure=read(folder/'closed.json'); assert closure['passed']
        for name,wanted in closure['files'].items(): assert pin(folder/name)==wanted,name
    assert pin(TOOLS/'counter.py')==read(PROOF/'closed.json')['local_inputs']['counter.py']
    for name in ['checks.py','il_normalization.py']:
        assert pin(TOOLS/name)==pin(OLD/'collected/tools'/name)


def prepare():
    previous_closed(); assert not BASE.exists(); BASE.mkdir()
    bundle=BASE/'bundle';bundle.mkdir(); originals={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for p in TOOLS.iterdir():
        if p.is_file():
            if p.suffix=='.py':ast.parse(p.read_text(encoding='utf8'),str(p))
            originals[p.relative_to(ROOT).as_posix()]=pin(p)
    for name in ['protocol.py','remote.py','remote_base.py','remote_prepare.py','counter.py','checks.py','il_normalization.py']:
        copy(TOOLS/name,bundle/'tools'/name)
    copy(TOOLS/'README.md',bundle/'README.md')
    shutil.copy2(ROOT/'.agent/m79-e5-frequency-observation-20260925.md',bundle/'prospective-plan.md')
    old=read(OLD/'payload.json'); receipt=read(OLD/'collected/collection.json');built=read(OLD/'collected/built.json')
    for source,name in [(OLD/'closed.json','original-closed.json'),(OLD/'collected/collection.json','original-collection.json'),
                        (OLD/'analysis.json','original-analysis.json'),(PROOF/'closed.json','clock-closed.json')]:
        copy(source,bundle/'evidence'/name)
    copy(OLD/'collected/cases-candidate.json',bundle/'cases-candidate.json')
    copy(OLD/'collected/evidence/original-e5-512tok-candidate.json',bundle/'evidence/original-e5-512tok-candidate.json')
    links={}
    for name,wanted in {**old['files'],**built['files']}.items():
        if name.startswith(('runtimes/candidate/','export-runtime/','reference/','tracer/')):
            links[name]=dict(source=REMOTE_OLD+'/'+name,identity=wanted)
    # Preserve actual original source and compiled review without building again.
    for name in ['source/global.json','source/consumer/Program.cs','source/consumer/ClockProbe.cs',
                 'source/consumer/NpySupport.cs','source/consumer/Producer.csproj',
                 'consumer-inventory/instructions.json','consumer-inventory/review.json',
                 'evidence/OriginalProgram.cs.txt','evidence/source-review.json']:
        copy(OLD/'collected'/name,bundle/name)
    built['files']={n:v for n,v in built['files'].items() if n in links}
    save(bundle/'built.json',built)
    external={n:v for n,v in old['external'].items() if n.startswith((
        '/home/vermorel/.dotnet/shared/Microsoft.NETCore.App/10.0.8/',
        '/home/vermorel/.dotnet/host/fxr/',
        '/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python/'))
        or n in ['/home/vermorel/.dotnet/dotnet','/home/vermorel/Onnx/models/multilingual-e5-small/model.onnx']}
    max_trace=max(v['bytes'] for n,v in receipt['files'].items() if n.endswith('capture.nettrace'))
    max_gzip=max(v['bytes'] for n,v in receipt['files'].items() if n.endswith('events.jsonl.gz'))
    copied=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file())
    physical=copied+2*(max_trace+max_gzip)+32*1024**2
    assert physical<128*1024**2
    logical=physical+sum(v['identity']['bytes'] for v in links.values())
    assert logical<LIMITS['artifacts']
    save(bundle/'stage.json',dict(passed=True,links=links,receipts=dict(original=REMOTE_OLD+'/collection.json'),
        external=external,products=dict(candidate=old['products']['candidate']),interpreter=old['interpreter'],
        previous_consumer=old['previous_consumer'],failed_release_controls=old['failed_release_controls'],
        clock_proof=pin(PROOF/'closed.json'),original_closure=pin(OLD/'closed.json'),
        storage_estimate=dict(physical_bytes=physical,logical_bytes=logical,initial_reserve=128*1024**2,
                              prior_max_trace=max_trace,prior_max_compressed=max_gzip,allowance=32*1024**2),
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()}))
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(passed=True,archive=pin(BASE/'payload.tar.gz'),links=len(links),physical_estimate=physical,logical_estimate=logical)))


if __name__=='__main__':prepare()
