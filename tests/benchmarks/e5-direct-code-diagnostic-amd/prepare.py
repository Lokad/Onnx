"""Reuse exact closed products and inputs; add only targeted code logging."""
import ast
import json
from pathlib import Path
import shutil
import tarfile
from protocol import LIMITS, DIAGNOSTIC_FLAGS, pin,read,save
from source_scope import instrument,verify
ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/e5-direct-code-diagnostic-amd-20260925'
PARENT=ROOT/'artifacts/e5-direct-tier-diagnostic-v2-amd-20260925'
REMOTE_PARENT='/dev/shm/lokad-e5-direct-tier-diagnostic-v2-20260925'


def previous_closed():
    assert pin(PARENT/'closed.json')['sha256']=='e61941d9308e33b688f222665a0f79dd985f87b85d56123554a47202d8c531bc'
    closure=read(PARENT/'closed.json');assert closure['passed'] and closure['diagnostic_only'] and not closure['release_admitted']
    for name in ['analysis.json','payload.json','collected/collection.json','collected/built.json']:
        assert pin(PARENT/name)==closure['files'][name],name
    receipt=read(PARENT/'collected/collection.json')
    assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
    assert read(PARENT/'collected/built.json')['consumer']==dict(bytes=36864,sha256='968d3beb8e19daa635af0c0e051f41f7f586e0a77a5a84120de5ba56c8c8b843')
    return closure


def prepare():
    assert not BASE.exists();closure=previous_closed();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={}
    files=read(PARENT/'collected/collection.json')['files'];parent=read(PARENT/'payload.json')
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    def parent_copy(name,target):
        source=PARENT/'collected'/name;assert pin(source)==files[name]
        copy(source,target)
    for name in ['ClockProbe.cs','NpySupport.cs','Producer.csproj']:
        parent_copy('source/consumer/'+name,bundle/'source/consumer'/name)
    parent_copy('source/global.json',bundle/'source/global.json')
    parent_copy('source/consumer/Program.cs',bundle/'evidence/OriginalProgram.cs.txt')
    original=(bundle/'evidence/OriginalProgram.cs.txt').read_text();actual=instrument(original)
    (bundle/'source/consumer/Program.cs').write_text(actual)
    copy(TOOLS/'DisassemblyPolicy.cs',bundle/'source/consumer/DisassemblyPolicy.cs')
    save(bundle/'evidence/source-review.json',verify(original,actual))
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py','il_normalization.py','remote_base.py']:
        copy(TOOLS/name,bundle/'tools'/name)
    original_supervisor=(ROOT/'tests/parakeet/dispatch-events-amd/remote.py').read_text()
    modified=(TOOLS/'remote_base.py').read_text()
    changes=[('from protocol import LIMITS,PROVIDERS,check_sample,pin,read,save,verify','from protocol import LIMITS,PROVIDERS,DIAGNOSTIC_FLAGS,check_sample,pin,read,save,verify'),
        ("command,build,cpu=command_for(name,spec);spawn('worker',command,build_env if build else env,cpu)",
         "command,build,cpu=command_for(name,spec);job_env=build_env if build else dict(env,**DIAGNOSTIC_FLAGS) if name.endswith('-capture') else env;spawn('worker',command,job_env,cpu)")]
    for old,new in reversed(changes):
        assert modified.count(new)==1;modified=modified.replace(new,old)
    assert modified==original_supervisor
    copy(ROOT/'tests/parakeet/dispatch-events-amd/remote.py',bundle/'evidence/OriginalSupervisor.py.txt')
    save(bundle/'evidence/supervisor-review.json',dict(passed=True,original_recovered_exactly=True,worker_only_logging=DIAGNOSTIC_FLAGS))
    copy(TOOLS/'README.md',bundle/'README.md')
    shutil.copy2(ROOT/'.agent/e5-direct-code-diagnostic-20260925.md',bundle/'prospective-plan.md')
    copy(PARENT/'closed.json',bundle/'evidence/parent-closed.json')
    copy(PARENT/'collected/collection.json',bundle/'evidence/parent-collection.json')
    for name in ['consumer-inventory/review.json','observer-inventory/review.json','built.json']:
        parent_copy(name,bundle/'evidence'/('parent-'+name.replace('/','-')))
    parent_copy('evidence/graph-analysis.json',bundle/'evidence/graph-analysis.json')
    links={}
    for role in ['current','candidate']:
        parent_copy(f'cases-{role}.json',bundle/f'cases-{role}.json')
        assert read(bundle/f'cases-{role}.json')['core']==parent['products'][role]['Lokad.Onnx.dll']['sha256']
        parent_copy(f'evidence/original-e5-8tok-{role}.json',bundle/f'evidence/original-e5-8tok-{role}.json')
        for name,wanted in files.items():
            if name.startswith(f'runtimes/{role}/') and not Path(name).name.startswith('ReleaseBenchmark.'):
                links[name]=dict(source=REMOTE_PARENT+'/'+name,identity=wanted)
        case,=read(bundle/f'cases-{role}.json')['cases'];assert case['key']=='e5-8tok'
        for row in [*case['inputs'],*case['outputs']]:
            if 'file' in row:
                name=row['file'];links[name]=dict(source=REMOTE_PARENT+'/'+name,identity=parent['files'][name])
    for name,wanted in parent['files'].items():
        if name.startswith(('tracer/','bridge/','export-runtime/')):
            links[name]=dict(source=REMOTE_PARENT+'/'+name,identity=wanted)
    for name,wanted in files.items():
        if name.startswith('runtimes/current/'):
            links[name.replace('runtimes/current/','previous/')]=dict(source=REMOTE_PARENT+'/'+name,identity=wanted)
    raw_max=max(files[f'{r}-capture/capture.nettrace']['bytes'] for r in 'abcd')
    compressed_max=max(files[f'{r}-export/events/events.jsonl.gz']['bytes'] for r in 'abcd')
    input_bytes=sum(r['identity']['bytes'] for r in links.values())+sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file())
    estimate=input_bytes+4*(raw_max+compressed_max)+96*1024**2
    assert estimate<LIMITS['artifacts']
    save(bundle/'stage.json',dict(passed=True,parent_closure=pin(PARENT/'closed.json'),links=links,
        receipts={'parent':REMOTE_PARENT+'/collection.json'},products=parent['products'],external=parent['external'],
        previous_consumer=read(PARENT/'collected/built.json')['consumer'],reused_exporter=parent['reused_exporter'],
        feed=parent['feed'],interpreter=parent['interpreter'],failed_release_cases=parent['failed_release_cases'],
        storage_estimate=dict(bytes=estimate,input_bytes=input_bytes,previous_max_trace=raw_max,previous_max_compressed=compressed_max,allowance=96*1024**2),
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for path in TOOLS.iterdir():
        if path.is_file():
            if path.suffix=='.py':ast.parse(path.read_text(),str(path))
            originals[path.relative_to(ROOT).as_posix()]=pin(path)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file():archive.add(path,arcname=path.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(passed=True,archive=pin(BASE/'payload.tar.gz'),links=len(links),estimated_bytes=estimate)))


if __name__=='__main__':prepare()
