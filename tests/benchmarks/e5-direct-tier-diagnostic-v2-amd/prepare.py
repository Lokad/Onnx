"""Bind the exact release and direct-depthwise candidate to one short-e5 input."""
import ast
import json
import math
from pathlib import Path
import shutil
import tarfile
from protocol import CALLS, LIMITS, pin, read, save
from source_scope import instrument, verify

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/e5-direct-tier-diagnostic-v2-amd-20260925'
GRAPH=ROOT/'artifacts/parakeet-packed-final-row-graphs-amd-20260925'
TRACE=ROOT/'artifacts/e5-repeatability-diagnostic-amd-20260925'
APP=ROOT/'artifacts/parakeet-direct-depthwise-app-amd-20260925'
REMOTE_GRAPH='/dev/shm/lokad-parakeet-packed-final-row-graphs-20260925'
REMOTE_TRACE='/dev/shm/lokad-e5-repeatability-diagnostic-20260925'
REMOTE_APP='/dev/shm/lokad-parakeet-direct-depthwise-app-20260925'
CLOSURES={
    GRAPH:'0b82805aa6de28287c9bc9242416b3abd5c58b76fdff57a0416208f29f3707f2',
    TRACE:'fe8ec805d18864a3a8d8b5a052543a3eef466535adc2769efc8c6c6747338a7b',
    APP:'e931b8b165a3f2a7cb5ff3f057de8378b42c7133e6a0637aad3207cf5da68e0a',
}


def previous_closed():
    rejected=ROOT/'artifacts/e5-direct-tier-diagnostic-amd-20260925'
    assert pin(rejected/'closed.json')['sha256']=='d1d534621e336b81f404bfcc2d7bf2ae81b1fec4512a50fa9d849f20e33b6d92'
    closure=read(rejected/'closed.json');assert closure['evidence_validated'] and not closure['campaign_passed']
    for name in ['failure-analysis.json','collected/collection.json']:
        assert pin(rejected/name)==closure['files'][name]
    for folder,digest in CLOSURES.items():
        assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');assert proof['passed']
        for name in ['analysis.json','payload.json','collected/collection.json']:
            assert pin(folder/name)==proof['files'][name],name
        receipt=read(folder/'collected/collection.json')
        assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
    failed=[r for r in read(GRAPH/'analysis.json')['performance'] if not r['regression_passed']]
    assert [r['key'] for r in failed]==['e5-8tok']
    assert failed[0]['candidate_over_current']>1.05
    return failed


def prepare():
    assert not BASE.exists();failed=previous_closed();BASE.mkdir()
    bundle=BASE/'bundle';bundle.mkdir();originals={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for name in ['ClockProbe.cs','Producer.csproj']:
        copy(TOOLS/name,bundle/'source/consumer'/name)
    old_probe=(TOOLS.parent/'e5-repeatability-diagnostic-amd/ClockProbe.cs').read_text()
    assert (TOOLS/'ClockProbe.cs').read_text()==old_probe.replace('Calls.Count != 780','Calls.Count != 6000')
    originals['tests/benchmarks/e5-repeatability-diagnostic-amd/ClockProbe.cs']=pin(TOOLS.parent/'e5-repeatability-diagnostic-amd/ClockProbe.cs')
    source=ROOT/'tests/benchmarks/warmed-release-amd-v2/Program.cs'
    copy(source,bundle/'evidence/OriginalProgram.cs.txt');original=source.read_text();actual=instrument(original)
    (bundle/'source/consumer/Program.cs').write_text(actual,encoding='utf8')
    save(bundle/'evidence/source-review.json',verify(original,actual))
    copy(ROOT/'tests/benchmarks/release-amd-v2/NpySupport.cs',bundle/'source/consumer/NpySupport.cs')
    copy(ROOT/'global.json',bundle/'source/global.json')
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py','il_normalization.py']:
        copy(TOOLS/name,bundle/'tools'/name)
    copy(ROOT/'tests/parakeet/dispatch-events-amd/remote.py',bundle/'tools/remote_base.py')
    copy(TOOLS/'README.md',bundle/'README.md')
    shutil.copy2(ROOT/'.agent/e5-direct-tier-diagnostic-20260925.md',bundle/'prospective-plan.md')
    receipts={}
    for label,folder,remote in [('graph',GRAPH,REMOTE_GRAPH),('trace',TRACE,REMOTE_TRACE),('application',APP,REMOTE_APP)]:
        for suffix in ['closed','collection']:
            copy(folder/('closed.json' if suffix=='closed' else 'collected/collection.json'),bundle/'evidence'/f'{label}-{suffix}.json')
        receipts[label]=remote+'/collection.json'
    rejected=ROOT/'artifacts/e5-direct-tier-diagnostic-amd-20260925'
    copy(rejected/'closed.json',bundle/'evidence/rejected-closed.json')
    copy(rejected/'failure-analysis.json',bundle/'evidence/rejected-analysis.json')
    copy(rejected/'collected/collection.json',bundle/'evidence/rejected-collection.json')
    receipts['rejected']='/dev/shm/lokad-e5-direct-tier-diagnostic-20260925/collection.json'
    copy(GRAPH/'analysis.json',bundle/'evidence/graph-analysis.json')
    for name in ['built.json','exporter-roundtrip/review.json','exporter-roundtrip/events/summary.json']:
        assert pin(TRACE/'collected'/name)==read(TRACE/'collected/collection.json')['files'][name]
        copy(TRACE/'collected'/name,bundle/'evidence'/('previous-'+Path(name).name))
    graph_payload=read(GRAPH/'payload.json');trace_payload=read(TRACE/'payload.json')
    graph_files=read(GRAPH/'collected/collection.json')['files'];trace_files=read(TRACE/'collected/collection.json')['files']
    app_files=read(APP/'collected/collection.json')['files']
    products={'current':graph_payload['products']['current'],
              'candidate':{'Lokad.Onnx.dll':read(APP/'payload.json')['identities']['candidate']['Lokad.Onnx.dll']}}
    assert products['current']['Lokad.Onnx.dll']['sha256']=='f95a13c58354bf07f3b7926b72903c18b1a560a56673297cb9fe001d3541b592'
    assert products['candidate']['Lokad.Onnx.dll']['sha256']=='40260aef7fd93c5153601ec104a87843a2c017a2720fd64c9e24e3460d455749'
    links={}
    # Only SDK/runtime, the offline feed and the single model are external inputs.
    external={n:v for n,v in trace_payload['external'].items()
              if n.startswith('/home/vermorel/.dotnet/') or n.startswith(trace_payload['feed']+'/')}
    for role in ['current','candidate']:
        manifest=read(GRAPH/f'collected/cases-{role}.json');case,=[c for c in manifest['cases'] if c['key']=='e5-8tok']
        manifest['cases']=[case];manifest['core']=products[role]['Lokad.Onnx.dll']['sha256']
        save(bundle/f'cases-{role}.json',manifest)
        copy(GRAPH/f'collected/timing-e5-8tok-{role}-a/output/result.json',bundle/f'evidence/original-e5-8tok-{role}.json')
        for name,wanted in graph_files.items():
            if name.startswith(f'runtimes/{role}/') and not Path(name).name.startswith('ReleaseBenchmark.'):
                if name.endswith('/Lokad.Onnx.dll') and role=='candidate':
                    source='runtimes/candidate/Lokad.Onnx.dll'
                    assert app_files[source]==products[role]['Lokad.Onnx.dll']
                    links[name]=dict(source=REMOTE_APP+'/'+source,identity=app_files[source])
                else:links[name]=dict(source=REMOTE_GRAPH+'/'+name,identity=wanted)
        external[case['model']]=graph_payload['external'][case['model']]
        for row in [*case['inputs'],*case['outputs']]:
            if 'file' in row:
                name=row['file'];links[name]=dict(source=REMOTE_GRAPH+'/'+name,identity=graph_payload['files'][name])
    for prefix in ['tracer/','bridge/','export-runtime/']:
        catalog=trace_payload['files'] if prefix=='tracer/' else trace_files
        for name,wanted in catalog.items():
            if name.startswith(prefix):links[name]=dict(source=REMOTE_TRACE+'/'+name,identity=wanted)
    for folder,remote,catalog in [('previous',REMOTE_GRAPH,graph_files),('previous-observer',REMOTE_TRACE,trace_files)]:
        for name,wanted in catalog.items():
            if name.startswith('runtimes/current/'):
                links[name.replace('runtimes/current/',folder+'/')]=dict(source=remote+'/'+name,identity=wanted)
    previous_built=read(TRACE/'collected/built.json')
    exporter=dict(binary=previous_built['exporter'],closure=pin(TRACE/'closed.json'),
                  roundtrip=read(TRACE/'collected/exporter-roundtrip/review.json'))
    assert exporter['roundtrip']['passed']
    raw_max=max(trace_files[f'{r}-capture/capture.nettrace']['bytes'] for r in 'abcd')
    compressed_max=max(trace_files[f'{r}-export/events/events.jsonl.gz']['bytes'] for r in 'abcd')
    input_bytes=sum(r['identity']['bytes'] for r in links.values())+sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file())
    estimate=input_bytes+math.ceil(4*CALLS/780*(raw_max+compressed_max))+64*1024**2
    assert estimate<LIMITS['artifacts'],(estimate,LIMITS['artifacts'])
    save(bundle/'stage.json',dict(passed=True,links=links,receipts=receipts,products=products,external=external,
        previous_consumer=graph_files['runtimes/current/ReleaseBenchmark.dll'],
        previous_observer=trace_files['runtimes/current/ReleaseBenchmark.dll'],reused_exporter=exporter,
        feed=trace_payload['feed'],interpreter=trace_payload['interpreter'],failed_release_cases=failed,
        storage_estimate=dict(bytes=estimate,input_bytes=input_bytes,previous_short_max_trace=raw_max,
            previous_short_max_compressed=compressed_max,call_ratio=CALLS/780,allowance=64*1024**2),
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
