"""Preserve the first resource-limited attempt; do not turn a prefix into a verdict."""
from pathlib import Path
import json,shutil
import numpy as np
from common import ROOT,pin,read,write,verify
from audit import absent,source,validate_record

base=ROOT/'artifacts/whisper-input-cross-20260920'
assert not (base/'failed-closed.json').exists()
spec=read(base/'manifest.json');verify(spec)
state=read(base/'managed-process/identity.json');result=read(base/'managed/result.json')
samples=[json.loads(line) for line in (base/'managed-process/samples.jsonl').read_text().splitlines()]
assert state['complete'] is False and state['code']==2 and state['terminal_members'] is True
assert 'Resource guard' in state['error'] and result['complete'] is False
assert state['manifest_sha256']==result['manifest_sha256']==pin(base/'manifest.json')['sha256']
assert len(samples)==state['samples']==350 and all(a['seconds']<b['seconds'] for a,b in zip(samples,samples[1:]))
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
assert members==state['members'] and max(totals)==state['peak_rss']==8797216768
assert max(totals[:-1])<=spec['limits']['rss']<totals[-1]
births=[state['supervisor']]+[dict(pid=int(pid),birth=birth) for pid,birth in members.items()]
assert all(absent(i) for i in births)
assert len(result['records'])==9 and not (base/'native-process').exists() and not (base/'native').exists()
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
# Copy exact original sources before any later experiment changes working files.
snapshots={}
for name,want in spec['files'].items():
    if name.startswith('tests/'):
        destination=base/'frozen-source'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        assert not destination.exists();shutil.copyfile(ROOT/name,destination);assert pin(destination)==want
        snapshots[name]=dict(file=destination.relative_to(base).as_posix(),**want)
report=Path(__file__).with_name('resource-failure-20260920.md');assert not report.exists()
report.write_text('''# Whisper crossed-input attempt: resource limit — 2026-09-20

The first encoder-only attempt stopped at its fixed **8 GiB process-group RSS
limit**. The triggering sample was **8,797,216,768 bytes** at 177.83 seconds.
All 350 resource samples are retained. Time and available-memory guards passed;
the RSS guard stopped only the owned process group. The supervisor and both
observed children are confirmed absent. No native worker started.

Nine of the planned 84 encoder outputs completed: five managed baseline bridges
and four managed-encoder/native-feature outputs. All five bridges reproduce the
saved baseline bytes exactly. Every completed output, shape, input hash, numerical
count and held-output record was independently checked. This incomplete prefix
does **not** establish the planned full-corpus input/engine decomposition.

The probe created a fresh execution context for each encoder call. Source review
confirms that the original public transcriber also creates fresh contexts per
request. Explicit contexts have independent released-buffer caches, but this run
does not distinguish retained caches, temporary allocations and GC behavior as
causes of the rising RSS. It is not evidence of an unbounded production leak.

The 8 GiB guard remains unchanged and this failed attempt will not be replayed
unchanged. A future bounded diagnostic must declare its execution-context or
process-lifetime change before inference and require all baseline bridges again.
No product code, numerical threshold, benchmark ratio or qualification changes.

Original tool source is `9571a6a`. Manifest SHA-256:
`0818f1a10c27ea525d63b87aa0e66f792a40d6da9be0988d6bac11fe45b05e2e`.
The complete failure receipt, full completed arrays, resource samples and exact
source snapshots are retained in `artifacts/whisper-input-cross-20260920`.
''',encoding='utf-8')
files={p.relative_to(base).as_posix():pin(p) for p in sorted(base.rglob('*')) if p.is_file()}
write(base/'failed-closed.json',dict(schema=1,status='execution_failed_resource_guard',complete_experiment=False,
    completed_arrays=9,baseline_bridges=5,native_started=False,all_births_terminal=True,births=births,peak_rss=state['peak_rss'],
    limit_rss=spec['limits']['rss'],samples=len(samples),files=files,source_snapshots=snapshots,
    closure_tool=dict(file=Path(__file__).relative_to(ROOT).as_posix(),**pin(Path(__file__))),
    report=dict(file=report.relative_to(ROOT).as_posix(),**pin(report))))
print('Failed attempt closed:',pin(base/'failed-closed.json'))
