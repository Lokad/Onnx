"""Audit a resource-guard failure without treating a saved prefix as completed endurance."""
from pathlib import Path
import copy,datetime,json,math,shutil
from deploy import BASE,ROOT
from close import terminal
from protocol import pin,read,write,LIMITS,check_sample,check_result,validate_records
from reuse_protocol import validate_reuse,allocation_gate
from audit import old


def prefix(rows,manifest):
    assert 0<len(rows)<80
    previous=0
    for index,row in enumerate(rows):
        case=manifest['cases'][index%20]
        assert row['name']==case['name'] and row['pass']==index//20 and row['phase']==('warmup' if index<20 else 'measured')
        assert row['input_sha256']==case['raw_sha256'] and row['ownership'] is True
        assert check_result(row['result'],case['expected'],family='whisper')==row['maximum_centroid_error']==0
        assert all(type(row[k]) is int for k in ['start_ticks','end_ticks','frequency'])
        assert previous<=row['start_ticks']<row['end_ticks'] and row['frequency']>0;previous=row['end_ticks']
        assert math.isfinite(row['seconds']) and row['seconds']>0
        assert math.isclose(row['seconds'],(row['end_ticks']-row['start_ticks'])/row['frequency'],rel_tol=1e-14)
    # This envelope checks only saved telemetry, never worker completion/identity.
    validate_reuse(dict(diagnostic='bounded-context-reuse-no-forced-gc',records=rows))


def main():
    base=BASE/'collected';receipt=read(base/'collection.json');state=read(base/'campaign/identity.json');frozen=read(base/'frozen.json')
    assert receipt['terminal'] and receipt['code']==state['code']==1 and state['complete']
    assert state['error']=='AssertionError()' and len(state['runs'])==2
    assert state['limits']==LIMITS and state['frozen']==pin(base/'frozen.json')==pin(BASE/'frozen.json')
    assert frozen['scope']=='whisper-buffer-reuse' and frozen['explicit_gc'] is False
    for name,wanted in receipt['files'].items():assert pin(base/name)==wanted,name
    for name,wanted in frozen['files'].items():assert pin(base/name)==wanted,name
    terminal(receipt['births']);manifest=read(base/'manifests/whisper.json')
    complete,failed=state['runs'];assert complete['phase']=='conformance' and failed['phase']=='timing'
    assert all((r['family'],r['engine'])==('whisper','managed') for r in [complete,failed])
    complete_dir=base/complete['output'];value=read(complete_dir/'worker/result.json')
    validate_records(value,manifest,'conformance');validate_reuse(value)
    assert value['runtime']=='.NET 10.0.8' and value['processor_count']==1 and value['flags']=={}
    assert value['manifest_sha256']==pin(base/'manifests/whisper.json')['sha256']
    for key,name in [('core_sha256','Lokad.Onnx.dll'),('data_sha256','Lokad.Onnx.Data.dll'),('runner_sha256','WhisperBufferReuse.dll')]:assert value[key]==pin(base/'bin'/name)['sha256']
    resources=old.resource_records(complete,[json.loads(s) for s in (complete_dir/'samples.jsonl').read_text().splitlines()])
    for i,row in enumerate(value['records']):assert read(complete_dir/'worker'/f'{i:03}.json')==row
    original=[read(base/'original-prefix'/f'{i:03}.json') for i in range(16)]
    gate=allocation_gate(value,original);gate.update(worker=pin(complete_dir/'worker/result.json'),frozen=pin(base/'frozen.json'))
    assert gate==read(base/'campaign/conformance-gate.json')
    assert failed['complete'] and failed['code'] is None and failed['error']=='AssertionError()'
    folder=base/failed['output'];assert not (folder/'worker/result.json').exists()
    rows=[read(p) for p in sorted((folder/'worker').glob('[0-9][0-9][0-9].json'))];prefix(rows,manifest)
    assert {p.name for p in (folder/'worker').iterdir()}=={f'{i:03}.json' for i in range(len(rows))}
    samples=[json.loads(s) for s in (folder/'samples.jsonl').read_text().splitlines()];assert len(samples)==failed['samples'] and samples
    for row in samples[:-1]:check_sample(row)
    last=samples[-1];old.refuses(lambda:check_sample(last));violations=[]
    if last['available']<LIMITS['available']:violations.append('available-memory reserve')
    if sum(m['rss'] for m in last['members'])>=LIMITS['rss']:violations.append('RSS limit')
    assert violations
    assert all(m['rss']>=0 for m in last['members'])
    check_sample(dict(last,available=max(last['available'],LIMITS['available']),members=[dict(m,rss=0) for m in last['members']]))
    assert failed['preflight_available']>=LIMITS['preflight'] and failed['preflight_disk']>=LIMITS['preflight_disk']
    assert 0<failed['seconds']<LIMITS['seconds'] and failed['peak_rss']==max(sum(m['rss'] for m in s['members']) for s in samples)
    previous=0;seen={};gaps=[]
    for sample in samples:
        assert previous<sample['seconds']<failed['seconds'];gaps.append(sample['seconds']-previous);previous=sample['seconds']
        for member in sample['members']:
            assert member['birth']==failed['members'][str(member['pid'])]>=failed['child']['birth'];seen[str(member['pid'])]=member['birth']
    gaps.append(failed['seconds']-previous);assert max(gaps)<10 and seen==failed['members']
    refusals=1
    for damage in [lambda r:r.pop(0),lambda r:r[0].update(ownership=False),lambda r:r[0].update(input_sha256='0'*64),
        lambda r:r[0]['result']['token_ids'].pop(),lambda r:r[1]['pools']['encodingExecution'].update(allocated_new_bytes=16*1024**2+1),
        lambda r:r[0]['memory_after'].update(ticks=0)]:
        bad=copy.deepcopy(rows);damage(bad);old.refuses(lambda:prefix(bad,manifest));refusals+=1
    result=dict(closure_passed=True,campaign_passed=False,prototype_only=True,benchmark=False,reason=violations,
        conformance=dict(calls=20,gate=gate,resource=resources,result=pin(complete_dir/'worker/result.json')),
        endurance=dict(completed_calls=len(rows),planned_calls=80,final_worker_result_exists=False,records=rows,resource_samples=len(samples),
            peak_rss=failed['peak_rss'],min_available=min(s['available'] for s in samples),last_sample=last,max_gap=max(gaps)),
        damaged_records_rejected=refusals,births=receipt['births'],frozen=pin(base/'frozen.json'),collection=pin(base/'collection.json'))
    write(BASE/'failure-audit.json',result);tracked=Path(__file__).resolve().parent
    write(tracked/'resource-failure-observations-20260920.json',result)
    text=f'''# Whisper buffer reuse reduces allocation but fails endurance

The private prototype passes all twenty conformance requests and its prospective
allocation gate, then fails the separate eighty-request endurance worker after
**{len(rows)} completed requests**. Its final resource sample violates the
**{', '.join(violations)}**. It is not qualified for the declared normal-runtime
endurance workload and supplies no matched ORT timing result.

Requests 2–16 in conformance allocate {gate['prototype_allocated_bytes']:,} bytes
versus {gate['original_allocated_bytes']:,} in the original saved prefix, a
{(1-gate['ratio'])*100:.2f}% reduction. Warm encoder new pool payload is at most
{gate['encoder_warm_max']:,} bytes. Those useful observations remain valid within
their scope; passing twenty requests does not substitute for the failed eighty.

The failed worker's peak sampled RSS is **{failed['peak_rss']:,} bytes**. Its last
sample has **{last['available']:,} available bytes**, against a
**{LIMITS['available']:,}-byte reserve**, and the RSS guard is
{LIMITS['rss']:,} bytes. All preceding {len(samples)-1:,} samples pass every guard.
Time, disk and thread affinity checks still pass at termination. The supervisor
stops only its owned process; all original process identities are terminal before
collection and independently checked again before reporting.

All saved requests match the original transcript/token/stop/no-speech references,
with the recorded input and ownership checks. Every saved cache/GC/heap snapshot
and resource sample is retained. The partial worker has no final result or final
held-output check; this audit does not fabricate that completion. The complete
conformance worker separately passes its final checks and all 1,205 resource samples.
No explicit collection or runtime override is applied, no guard is relaxed and
no unchanged worker is restarted. Existing encoder/logit numerical failures remain.

Frozen SHA-256: `{pin(base/'frozen.json')['sha256']}`. The collection retains
{len(receipt['files'])} private/output files and rechecks {receipt['external_verified']:,}
external identities. The audit rejects {refusals} damaged resource/request/telemetry
records. [Complete observations](resource-failure-observations-20260920.json)
include the successful conformance gate and every failed-worker saved request.
Raw evidence is under `artifacts/whisper-buffer-reuse-20260920`.

The independent [allocation review](allocation-review-20260920.md) identifies
remaining decoder allocation; the separate [private weight-sharing proof](../weight-sharing/local-results-20260920.md)
removes 635.19 MB of duplicated initializer payload locally without model inference.
Neither finding repairs this failed run or establishes the next variant's AMD
application/resource behavior. Production source remains unchanged.
'''
    with (tracked/'resource-failure-20260920.md').open('x',encoding='utf-8') as f:f.write(text)
    benchmark=ROOT/'BENCHMARK.md';s=benchmark.read_text(encoding='utf-8')
    marker='20. It measures reclaimability and does not supply normal-runtime timing.\n';assert s.count(marker)==1
    s=s.replace(marker,marker+f'A [private buffer-reuse prototype](tests/whisper/buffer-reuse/resource-failure-20260920.md)\nreduces allocation and passes twenty conformance requests, but its separate\neighty-request endurance worker stops after {len(rows)} completed requests at a memory\nguard. AMD Whisper timing remains pending.\n');benchmark.write_text(s,encoding='utf-8')
    snapshots=BASE/'failure-tool-snapshots';snapshots.mkdir()
    for p in [benchmark,*sorted(tracked.glob('*.py'))]:
        target=snapshots/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    files={p.relative_to(ROOT).as_posix():pin(p) for p in sorted(BASE.rglob('*')) if p.is_file()}
    for name in ['resource-failure-20260920.md','resource-failure-observations-20260920.json']:files[(tracked/name).relative_to(ROOT).as_posix()]=pin(tracked/name)
    write(BASE/'failure-closed.json',dict(closure_passed=True,campaign_passed=False,closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,births=receipt['births']))
    for name,wanted in files.items():assert pin(ROOT/name)==wanted,name
    print(json.dumps(dict(closed=True,campaign_passed=False,conformance_calls=20,endurance_calls=len(rows),reason=violations,closure=pin(BASE/'failure-closed.json'))))


if __name__=='__main__':main()
