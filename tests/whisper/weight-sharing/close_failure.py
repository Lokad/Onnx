"""Retain the failed final storage assertion without fabricating a completed worker result."""
from pathlib import Path
import datetime,json,math,shutil,sys
from deploy import BASE,ROOT
from close import terminal
from protocol import pin,read,write,LIMITS,check_sample,check_result
sys.path.append(str(ROOT/'tests/whisper/buffer-reuse'))
sys.path.append(str(ROOT/'tests/whisper/memory-collection'))
from reuse_protocol import validate_reuse,allocation_gate


def main():
    base=BASE/'collected';receipt=read(base/'collection.json');state=read(base/'campaign/identity.json');frozen=read(base/'frozen.json')
    assert receipt['terminal'] and receipt['code']==state['code']==1 and state['complete'] and len(state['runs'])==1
    assert frozen['scope']=='whisper-weight-sharing' and frozen['explicit_gc'] is False
    assert state['frozen']==pin(base/'frozen.json')==pin(BASE/'frozen.json') and state['limits']==LIMITS
    for name,wanted in receipt['files'].items():assert pin(base/name)==wanted,name
    for name,wanted in frozen['files'].items():assert pin(base/name)==wanted,name
    terminal(receipt['births']);run=state['runs'][0];folder=base/run['output'];manifest=read(base/'manifests/whisper.json')
    assert run['phase']=='conformance' and run['code']==-6 and run['complete'] and run['error']=='AssertionError()'
    error=(folder/'stderr.txt').read_text();assert 'InvalidDataException: Decoder weights or storage changed' in error
    assert not (folder/'worker/result.json').exists() and not (base/'campaign/conformance-gate.json').exists()
    paths=sorted((folder/'worker').glob('[0-9][0-9][0-9].json'));assert len(paths)==20
    assert {p.name for p in (folder/'worker').iterdir()}=={f'{i:03}.json' for i in range(20)}
    rows=[read(p) for p in paths];previous=0
    for row,case in zip(rows,manifest['cases'],strict=True):
        assert row['name']==case['name'] and row['pass']==0 and row['phase']=='warmup' and row['ownership'] is True
        assert row['input_sha256']==case['raw_sha256'] and check_result(row['result'],case['expected'],family='whisper')==0
        assert previous<=row['start_ticks']<row['end_ticks'] and row['frequency']>0;previous=row['end_ticks']
        assert math.isclose(row['seconds'],(row['end_ticks']-row['start_ticks'])/row['frequency'],rel_tol=1e-14)
    envelope=dict(diagnostic='bounded-context-reuse-no-forced-gc',records=rows);validate_reuse(envelope)
    original=[read(base/'original-prefix'/f'{i:03}.json') for i in range(16)]
    allocation=allocation_gate(envelope,original)
    samples=[json.loads(s) for s in (folder/'samples.jsonl').read_text().splitlines()]
    assert len(samples)==run['samples'] and run['preflight_available']>=LIMITS['preflight'] and run['preflight_disk']>=LIMITS['preflight_disk']
    previous=0;seen={};gaps=[]
    for row in samples:
        check_sample(row);assert previous<row['seconds']<run['seconds'];gaps.append(row['seconds']-previous);previous=row['seconds']
        for member in row['members']:
            assert member['birth']==run['members'][str(member['pid'])]>=run['child']['birth'];seen[str(member['pid'])]=member['birth']
    gaps.append(run['seconds']-previous);assert max(gaps)<10 and seen==run['members']
    peak=max(sum(m['rss'] for m in row['members']) for row in samples);assert peak==run['peak_rss']
    result=dict(closure_passed=True,campaign_passed=False,reason='Final decoder snapshot equality assertion failed',
        final_worker_result_exists=False,endurance_started=False,saved_calls=20,records=rows,allocation_prefix=allocation,
        resource=dict(samples=len(samples),peak_rss=peak,min_available=min(r['available'] for r in samples),min_disk=min(r['disk'] for r in samples),max_gap=max(gaps)),
        frozen=pin(base/'frozen.json'),collection=pin(base/'collection.json'),births=receipt['births'],stderr=error)
    write(BASE/'failure-audit.json',result);tracked=Path(__file__).parent;report=tracked/'failure-20260920.md';data=tracked/'failure-observations-20260920.json';write(data,result)
    text=f'''# Whisper sharing candidate fails its final snapshot assertion

All twenty saved AMD conformance transcripts, token sequences, stop/no-speech
decisions, PCM and per-request ownership checks pass. All {len(samples):,} resource
samples pass; peak RSS is {peak:,} bytes. Matching calls 2–16 allocate
{allocation['prototype_allocated_bytes']:,} bytes versus the original
{allocation['original_allocated_bytes']:,} bytes.

The worker then throws **`Decoder weights or storage changed`** while comparing
complete pre/post decoder snapshots. It exits with code -6 and writes no final
result. The conditional eighty-request endurance phase correctly does not start.
This campaign is **not qualified**; the saved prefix does not substitute for its
failed final check or the unstarted endurance phase.

The diagnostic compared payload hashes, names, graph metadata and storage counts
in one equality assertion, but failed to save the two snapshots before asserting.
Consequently this evidence cannot distinguish changed payload bytes from changed
metadata or storage membership. It establishes a snapshot mismatch, not weight
corruption or a root cause. A distinct bounded diagnostic is required. The earlier
local preparation-only proof and separate buffer-reuse success remain unchanged.

No production source changed, no guard was relaxed, and the failed workload is
not restarted. All original process identities are terminal before collection and
again before reporting. Frozen SHA256: `{pin(base/'frozen.json')['sha256']}`.
Collection retains {len(receipt['files'])} files and checks
{receipt['external_verified']:,} external identities. [Complete observations](failure-observations-20260920.json)
retain all saved calls, resource totals and the exact exception. Raw evidence is
under `artifacts/whisper-weight-sharing-20260920`. No matched ORT timing is supplied.
'''
    with report.open('x',encoding='utf-8') as f:f.write(text)
    benchmark=ROOT/'BENCHMARK.md';s=benchmark.read_text(encoding='utf-8');marker='It reduces allocations but does not supply a matched AMD Whisper timing result.\n';assert s.count(marker)==1
    s=s.replace(marker,marker+'A separate [decoder-weight sharing candidate](tests/whisper/weight-sharing/failure-20260920.md)\ncompletes twenty saved requests but fails its final decoder snapshot check;\nits endurance phase does not start. This candidate is not yet qualified.\n');benchmark.write_text(s,encoding='utf-8')
    snapshots=BASE/'failure-snapshots';snapshots.mkdir()
    for p in [benchmark,Path(__file__)]:shutil.copyfile(p,snapshots/p.name)
    files={p.relative_to(ROOT).as_posix():pin(p) for p in sorted(BASE.rglob('*')) if p.is_file()}
    files.update({p.relative_to(ROOT).as_posix():pin(p) for p in [report,data]})
    write(BASE/'failure-closed.json',dict(closure_passed=True,campaign_passed=False,files=files,births=receipt['births'],closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    for name,wanted in files.items():assert pin(ROOT/name)==wanted,name
    print(json.dumps(dict(closed=True,campaign_passed=False,closure=pin(BASE/'failure-closed.json'),pins=len(files),resource=result['resource'])))


if __name__=='__main__':main()
