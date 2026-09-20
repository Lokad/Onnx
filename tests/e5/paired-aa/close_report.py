"""Close an audited A/A campaign, preserving a failed timing screen as a valid result."""
from pathlib import Path
import argparse,json,statistics,subprocess,time
from prepare_inputs import sha,pin,read,write_new

def distributions(records):
    """Describe every measured call without exclusions or changing timing gates."""
    result=[]
    for index in sorted({v['schedule']['case_index'] for v in records}):
        group=[v for v in records if v['schedule']['case_index']==index]
        for phase in ('paired','solo'):
            arms=[]
            for arm in (0,1):
                rows=[(r,v['frequency']) for v in group for r in v['measured'] if r['arm']==arm and r['phase']==phase]
                assert rows
                arms.append(dict(calls=len(rows),
                    execute_median_seconds=statistics.median(r['execute']/f for r,f in rows),
                    execute_maximum_seconds=max(r['execute']/f for r,f in rows),
                    request_median_seconds=statistics.median(r['request']/f for r,f in rows),
                    request_maximum_seconds=max(r['request']/f for r,f in rows),
                    mean_reset_seconds=statistics.mean((r['request']-r['execute'])/f for r,f in rows),
                    mean_allocated_bytes=statistics.mean(r['bytes'] for r,f in rows),
                    calls_with_gc_delta=sum(any(r[k]>0 for k in ('g0','g1','g2')) for r,f in rows),
                    collection_deltas=[sum(r[k] for r,f in rows) for k in ('g0','g1','g2')]))
            result.append(dict(name=group[0]['initial'][0]['name'],phase=phase,arms=arms))
    assert sum(a['calls'] for row in result for a in row['arms'])==sum(len(v['measured']) for v in records)
    return result

def close(base):
    assert not (base/'closed.json').exists()
    audit=read(base/'audit.json');assert audit['execution_passed'] is True
    assert audit['auditor_sha256']==sha(base/'payload/audit.py')
    assert audit['bundle_sha256']==sha(base/'payload/bundle.json') and audit['collection_sha256']==sha(base/'collected/collection.json')
    for location,manifest in [(base/'payload',read(base/'payload/bundle.json')),(base/'collected',read(base/'collected/collection.json'))]:
        for name,wanted in manifest['files'].items():assert pin(location/name)==wanted,name
    script='''from pathlib import Path
import json
items=ITEMS
for item in items:
 p=Path('/proc')/str(item['pid'])/'stat'
 if p.exists():assert int(p.read_text().split(') ',1)[1].split()[19])!=item['start'],item
print(json.dumps(dict(all_absent=True,identities=items)))
'''.replace('ITEMS',repr(audit['resources']['terminal_processes']))
    check=subprocess.run(['ssh','-i','C:/Users/JoannesVermorel/.ssh/id_onnx-bench.pem','-o','BatchMode=yes','vermorel@74.178.91.76','python3 -B -'],input=script,text=True,capture_output=True,check=True)
    root=Path(__file__).resolve().parents[3]
    for item in read(base/'storage-reclamation.json')['removed']:
        assert pin(Path(item['local']))=={k:item[k] for k in ('bytes','sha256')},item['local']
    record=dict(schema=1,closed=True,closed_at=time.time(),all_owned_processes_terminal=True,terminal=json.loads(check.stdout),
        audit_sha256=sha(base/'audit.json'),timing_screen_passed=audit['timing']['passed'],
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        files={p.relative_to(base).as_posix():pin(p) for p in sorted(base.rglob('*')) if p.is_file()},
        tracked={p.relative_to(root).as_posix():pin(p) for p in sorted(Path(__file__).parent.iterdir()) if p.is_file()})
    write_new(base/'closed.json',record)
    print(json.dumps(dict(closed_sha256=sha(base/'closed.json'),files=len(record['files']),timing_screen_passed=record['timing_screen_passed']),indent=2))

def report(base,destination):
    markdown=destination/'results-20260920.md';observations=destination/'observations-20260920.json'
    assert not markdown.exists() and not observations.exists()
    closed=read(base/'closed.json');assert closed['closed'] and closed['all_owned_processes_terminal']
    assert {p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file()}==set(closed['files'])|{'closed.json'}
    for name,wanted in closed['files'].items():assert pin(base/name)==wanted,name
    audit=read(base/'audit.json');assert audit['execution_passed'] and sha(base/'audit.json')==closed['audit_sha256']
    identity=read(base/'collected/result/identity.json')
    distribution=distributions([read(base/'collected/result'/row['name']/'result.json') for row in identity['runs']])
    value=dict(audit,distributions=distribution,closed_sha256=sha(base/'closed.json'),reporter_sha256=sha(Path(__file__)))
    write_new(observations,value)
    disposition=('passes its prospective timing screens. This permits designing a separate managed candidate comparison; it does not establish deployment equivalence or native ORT parity'
        if audit['timing']['passed'] else 'fails its prospective timing screens. It is not qualified to decide the intended small managed performance differences, and no candidate comparison follows from this result')
    text=f'''# Paired managed e5 A/A experiment — September 20, 2026

The twenty-worker A/A experiment {disposition}. Product code, defaults and the current e5/ORT scoreboard are unchanged. Both engines in every process use the same frozen core and public Memory option; no optimization candidate is present.

All complete native-output and bitwise A/A checks pass, including input and held-output ownership. The independent audit retains all {audit['measured_calls']:,} measured calls and {audit['conditioning_calls']:,} conditioning calls. Maximum scaled native output error is `{audit['maximum_error']:.12g}` at the unchanged `1e-4` limit.

| Case | A Execute ms | B Execute ms | B / A | Worker B/A range | AB/BA contrast | Paired/solo A / B | Execute screens |
|---|---:|---:|---:|---|---:|---|---|
'''
    for row in audit['timing']['observations']:
        r=row['boundaries']['execute']
        text+=f"| {row['name']} | {r['a_seconds']*1000:.4f} | {r['b_seconds']*1000:.4f} | {r['ratio']:.6f} | {min(r['visits']):.6f}–{max(r['visits']):.6f} | {r['order_contrast']:.6f} | {r['paired_solo'][0]:.6f} / {r['paired_solo'][1]:.6f} | {'pass' if all(r['gates'].values()) else 'fail'} |\n"
    text+='''
Every ratio should be close to one in A/A. Prospective limits are ±0.5% for aggregate B/A, ±1% for each worker and AB/BA contrast, and ±5% for each paired/solo ratio. Both timing boundaries must satisfy all limits. These screens use every sample; they are not confidence intervals or evidence of independent sampling. The worker ranges are descriptive ranges of four visits.

| Case | A Reset+Execute ms | B Reset+Execute ms | B / A | Worker B/A range | AB/BA contrast | Paired/solo A / B | Request screens |
|---|---:|---:|---:|---|---:|---|---|
'''
    for row in audit['timing']['observations']:
        r=row['boundaries']['request']
        text+=f"| {row['name']} | {r['a_seconds']*1000:.4f} | {r['b_seconds']*1000:.4f} | {r['ratio']:.6f} | {min(r['visits']):.6f}–{max(r['visits']):.6f} | {r['order_contrast']:.6f} | {r['paired_solo'][0]:.6f} / {r['paired_solo'][1]:.6f} | {'pass' if all(r['gates'].values()) else 'fail'} |\n"
    text+='''
The following descriptive table retains both paired and solo phases separately.
Every cell with two numbers gives A / B. The maximum includes every measured
call, including calls with a collection-count increment. Allocation and GC
observations describe this shared process; they do not isolate collector cost
or establish why a call was slow. Background collection can overlap a call
without a new count increment. No distribution statistic changes the screens.

| Case | Phase | Calls per arm | Execute median ms A / B | Execute maximum ms A / B | Calls with GC delta A / B | Mean allocated MB A / B |
|---|---|---:|---|---|---|---|
'''
    for row in distribution:
        a,b=row['arms'];assert a['calls']==b['calls']
        text+=f"| {row['name']} | {row['phase']} | {a['calls']} | {a['execute_median_seconds']*1000:.4f} / {b['execute_median_seconds']*1000:.4f} | {a['execute_maximum_seconds']*1000:.4f} / {b['execute_maximum_seconds']*1000:.4f} | {a['calls_with_gc_delta']} / {b['calls_with_gc_delta']} | {a['mean_allocated_bytes']/1e6:.6f} / {b['mean_allocated_bytes']/1e6:.6f} |\n"
    text+='\nComplete observations also retain request-boundary medians/maxima, mean Reset boundary overhead and generation-specific collection deltas for both phases. MB are decimal.\n'
    r=audit['resources']
    text+=f'''
The AMD EPYC 9V74 workers inherit CPU 2 before CLR startup, with supervisor CPU 0 and .NET 10.0.8. Two independent AssemblyLoadContexts keep product types and static fields separate; the runtime, collector and physical caches remain shared. Both graphs remain resident during solo blocks. Creation/case/phase order is balanced, and paired order uses a fixed seeded shuffle. See the [full protocol](README.md) for exact counts, ordering and reproduction.

Each engine gets thirty cumulative Execute seconds of conditioning, followed by sixty-four balanced adjacent pairs and thirty-two solo calls per arm. Public Execute and enclosing Reset-plus-Execute are timed on each call; loading, file IO, validation, output copies and serialization are excluded. No GC is forced, no sample is discarded, and no runtime override is set. Per-call allocation and GC counts are retained. This lane contains no native ORT timing and cannot update the managed/ORT ratio.

All resource limits pass. The maximum sampled process-group RSS is {r['peak_rss']/1e9:.6f} GB, minimum system available memory {r['minimum_available_memory']/1e9:.6f} GB, across {r['samples']:,} samples. The maximum observed foreign CPU fraction is {r['maximum_foreign_cpu_fraction']:.8g}; process snapshots miss some exited/short-lived activity. Guest CPU counters are also retained. These finite observations do not establish universal resource limits or control hypervisor neighbors.

Both projects build without warnings. Seven independent test methods exercise complete real smoke arrays, balanced/common-drift schedules, four synthetic bias scenarios, seven malformed schedules, nine damaged real results, twelve damaged resource records and six runtime guard failures. An initial smoke refused an Int64-versus-Int32 input-dimension encoding bug in the probe before model execution; that failure remains in the original artifact. The corrected smoke and its full native/ownership checks are separate. No product arithmetic or numerical gate changed.

The frozen source is {read(base/'payload/bundle.json')['source_commit']}; product source is `8732831`, core SHA `05884cfd524cc7130321f5dc1bcd0af17dddc7b97e8428d2d2f59e00edb795c2`. Four redundant remote result archives were removed only after verifying identical retained local backups; unique results/models and all local backups remain. The collection and independent audit are complete, and all observed process births are absent. Successful writers are closed and must not be rerun.

Raw evidence is under `artifacts/e5-paired-aa-v2-20260920`. [Complete observations](observations-20260920.json) preserve every screen disposition and evidence identity.

- Closed receipt: `{value['closed_sha256']}`.
- Payload bundle: `{audit['bundle_sha256']}`.
- Collection: `{audit['collection_sha256']}`.
'''
    markdown.write_text(text,encoding='utf-8');print('Rendered closed A/A disposition and observations.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('close','report'));p.add_argument('--artifact',type=Path,required=True);p.add_argument('--destination',type=Path,default=Path(__file__).parent)
    a=p.parse_args()
    if a.action=='close':close(a.artifact.resolve())
    else:report(a.artifact.resolve(),a.destination.resolve())
