"""Preserve the first resource-limited attempt; do not turn a prefix into a verdict."""
from pathlib import Path
import json,shutil,sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"input-cross"))
import numpy as np
from common import ROOT,pin,read,write,verify
from audit import absent,source,validate_record

base=ROOT/'artifacts/whisper-input-cross-reuse-20260920'
assert not (base/'failed-closed.json').exists()
spec=read(base/'manifest.json');verify(spec)
state=read(base/'managed-process/identity.json');result=read(base/'managed/result.json')
samples=[json.loads(line) for line in (base/'managed-process/samples.jsonl').read_text().splitlines()]
assert state['complete'] is False and state['code']==2 and state['terminal_members'] is True
assert 'Resource guard' in state['error'] and result['complete'] is False
assert state['manifest_sha256']==result['manifest_sha256']==pin(base/'manifest.json')['sha256']
assert len(samples)==state['samples']==472 and all(a['seconds']<b['seconds'] for a,b in zip(samples,samples[1:]))
assert state['preflight_available']>=spec['limits']['preflight_available']
members={};totals=[]
for sample in samples:
    assert 0<=sample['seconds']<spec['limits']['seconds'] and sample['available']>=spec['limits']['available']
    assert len({p['pid'] for p in sample['members']})==len(sample['members'])
    for process in sample['members']:
        assert process['affinity']==[2] and process['rss']>=0
        pid=str(process['pid']);birth=process['birth'];assert state['members'][pid]==birth
        assert pid not in members or members[pid]==birth;members[pid]=birth
    totals.append(sum(p['rss'] for p in sample['members']))
assert members==state['members'] and max(totals)==state['peak_rss']==8791007232
assert max(totals[:-1])<=spec['limits']['rss']<totals[-1]
births=[state['supervisor']]+[dict(pid=int(pid),birth=birth) for pid,birth in members.items()]
assert all(absent(i) for i in births)
assert len(result['records'])==12 and not (base/'native-process').exists() and not (base/'native').exists()
assert result['runtime']=='10.0.12' and result['affinity']==4 and result['processor_count']==1 and result['flags']=={} and result['native_loaded'] is False
assert result['core_sha256']==spec['core_sha256'] and result['probe_sha256']==pin(base/'bin/WhisperInputCross.dll')['sha256']
expected_files={'result.json'};first={}
for ordinal,row in enumerate(result['records']):
    index=ordinal//2;kind=['MM','MN'][ordinal%2];item=spec['requests'][index]
    name=f"{index:02}-{item['name']}-{kind}.f32";assert row['file']==name
    path=base/'managed'/name;assert pin(path)==dict(bytes=1500*1280*4,sha256=row['sha256'])
    value=np.fromfile(path,dtype='<f4').reshape(1,1500,1280)
    feature='managed_features' if kind=='MM' else 'native_features';source(item[feature])
    validate_record(row,index,item,kind,value,source(item['native_hidden']),item[feature]['raw_sha256'])
    if index==0:first[kind]=row['sha256']
    expected_files.add(name)
assert result['held_outputs']==first and {p.name for p in (base/'managed').iterdir()}==expected_files
assert result['context_lifecycle']=='one-reused-context'
previous=None
for row in result['records']:
    for memory in [row['memory_before'],row['memory_after']]:
        assert len(memory['collections'])==3 and all(type(x) is int and x>=0 for x in memory['collections'])
        assert all(type(memory[k]) is int and memory[k]>=0 for k in memory if k!='collections')
        assert memory['last_gc_fragmented']<=memory['last_gc_heap']<=memory['last_gc_committed']
        if previous:
            assert memory['allocated_total']>=previous['allocated_total'] and memory['gc_index']>=previous['gc_index']
            assert all(a>=b for a,b in zip(memory['collections'],previous['collections']))
        previous=memory
    assert row['pool_allocated_bytes']>=0 and row['pool_reused_bytes']>=0
    assert row['memory_after']['allocated_total']-row['memory_before']['allocated_total']>=row['pool_allocated_bytes']
assert all(row['memory_after']['collections']==[7,7,6] and row['memory_after']['gc_index']==7 for row in result['records'][1:])
# Copy exact original sources before any later experiment changes working files.
snapshots={}
for name,want in spec['files'].items():
    if name.startswith('tests/'):
        destination=base/'frozen-source'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        assert not destination.exists();shutil.copyfile(ROOT/name,destination);assert pin(destination)==want
        snapshots[name]=dict(file=destination.relative_to(base).as_posix(),**want)
report=Path(__file__).with_name('resource-failure-20260920.md');assert not report.exists()
report.write_text("""# Reused-context Whisper comparison: resource limit — 2026-09-20

The reused-context attempt also stopped at the unchanged **8 GiB process-group
RSS limit**, at **8,791,007,232 bytes** after 240.10 seconds. All 472 samples are
retained; time and available-memory guards passed. The minimum available memory
was 2,053,832,704 bytes. Only the owned process group was stopped; the supervisor
and both observed children are confirmed absent. No native worker started.

Twelve of 84 planned encoder outputs completed: six original managed baseline
bridges and six managed-encoder/native-feature outputs. All six bridges match
saved output bytes exactly. Every completed full array, input/shape/hash/gate
record, held output and allocation observation was independently checked. This
prefix does not establish the full twenty-recording input/engine decomposition.

One Memory GraphExecution was retained and Reset around every call. The changed
lifecycle was insufficient to complete the experiment within its bound. At the
second completed call, managed heap estimate was 3,467,724,816 bytes; at the
last it was 8,329,352,592 bytes. Collection counts remained [7, 7, 6], with most
recent GC index 7, throughout that interval. Allocated-byte counters rose by
approximately the same amount. These observations show allocation accumulation
since the last completed collection; they do not establish which objects remain
reachable, an unbounded leak, or the behavior of a complete public transcription.

The completed calls recorded 4,519,680,000 newly allocated pool payload bytes and
213,899,520,000 bytes served from pool reuse. Those are cumulative counters, not
peak or retained memory. The first fresh-context attempt remains a separately
closed failure. Neither attempt licenses a relaxed memory or numerical limit.

The next experiment will use finite worker lifetimes for each recording, retaining
the complete corpus and all baseline bridges. Process isolation changes the
execution lifetime, so it supplies numerical localization rather than a memory
or performance qualification of the public transcriber.

Original producer source: `2b67929`. Manifest SHA-256:
`90ac9ced9d1a094509f785d59bcc04b9c42ad9bfa6ac6d36d99f55178460491d`.
All arrays, resource samples, actual process births and exact source snapshots
are retained in `artifacts/whisper-input-cross-reuse-20260920`.
""",encoding='utf-8')
files={p.relative_to(base).as_posix():pin(p) for p in sorted(base.rglob('*')) if p.is_file()}
write(base/'failed-closed.json',dict(schema=1,status='execution_failed_resource_guard',complete_experiment=False,
    completed_arrays=12,baseline_bridges=6,native_started=False,all_births_terminal=True,births=births,peak_rss=state['peak_rss'],
    limit_rss=spec['limits']['rss'],samples=len(samples),files=files,source_snapshots=snapshots,
    closure_tool=dict(file=Path(__file__).relative_to(ROOT).as_posix(),**pin(Path(__file__))),
    report=dict(file=report.relative_to(ROOT).as_posix(),**pin(report))))
print('Failed attempt closed:',pin(base/'failed-closed.json'))
