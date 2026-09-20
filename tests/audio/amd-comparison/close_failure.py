"""Close the original all-family resource failure without inventing timing data."""
import datetime, hashlib, json
import numpy as np
from deploy import BASE, ROOT, ssh
from protocol import pin, read, write, LIMITS, validate_records, check_sample
from audit import resource_records, worker_identity


def main():
    base=BASE/'collected';receipt=read(base/'collection.json');state=read(base/'campaign/identity.json')
    assert receipt['terminal'] and receipt['code']==state['code']==1 and state['complete']
    assert state['error']=='AssertionError()' and len(state['runs'])==6
    for name,wanted in receipt['files'].items():assert pin(base/name)==wanted,name
    frozen=read(base/'frozen.json');assert state['frozen']==pin(base/'frozen.json')
    for name,wanted in frozen['files'].items():assert pin(base/name)==wanted,name
    observations=[]
    for run in state['runs'][:5]:
        folder=base/run['output'];manifest=read(base/'manifests'/(run['family']+'.json'))
        value=read(folder/'worker/result.json');validate_records(value,manifest,'conformance')
        worker_identity(value,manifest,frozen,base,run['engine'])
        samples=[json.loads(line) for line in (folder/'samples.jsonl').read_text().splitlines()]
        resources=resource_records(run,samples)
        for i,row in enumerate(value['records']):assert read(folder/'worker'/f'{i:03}.json')==row
        if run['family']=='whisper':
            for case,row in zip(manifest['cases'],value['records'],strict=True):
                a=np.load(folder/'worker'/(case['name']+'.features.npy'),allow_pickle=False)
                b=np.load(base/'assets'/case['features']['path'],allow_pickle=False)
                assert a.dtype==b.dtype==np.float32 and a.shape==b.shape==(1,128,3000)
                assert np.isfinite(a).all() and np.isfinite(b).all()
                d=np.abs(a.astype(np.float64)-b.astype(np.float64))
                assert row['frontend']==dict(values=a.size,max_abs=float(d.max()),failed=int((d>1e-5).sum()),bits_equal=a.tobytes()==b.tobytes(),sha256=hashlib.sha256(a.tobytes()).hexdigest())
        observations.append(dict(name=run['name'],calls=len(value['records']),resource=resources,result=pin(folder/'worker/result.json')))
    run=state['runs'][-1];assert run['name']=='conformance-05-whisper-managed' and run['complete'] and run['code'] is None
    folder=base/run['output'];assert not (folder/'worker/result.json').exists()
    samples=[json.loads(line) for line in (folder/'samples.jsonl').read_text().splitlines()]
    assert len(samples)==run['samples']==979
    for sample in samples[:-1]:check_sample(sample)
    last=samples[-1];assert last['available']<LIMITS['available']
    assert last['disk']>=LIMITS['disk'] and sum(m['rss'] for m in last['members'])<LIMITS['rss']
    repaired=dict(last,available=LIMITS['available']);check_sample(repaired)
    assert 0<run['seconds']<LIMITS['seconds'] and run['peak_rss']==max(sum(m['rss'] for m in s['members']) for s in samples)
    assert max(b['seconds']-a['seconds'] for a,b in zip(samples,samples[1:]))<10
    assert all(m['birth']==run['members'][str(m['pid'])] for s in samples for m in s['members'])
    records=[read(p) for p in sorted((folder/'worker').glob('[0-9]*.json'))]
    assert len(records)==16
    manifest=read(base/'manifests/whisper.json');prefix=dict(manifest,cases=manifest['cases'][:16])
    # This synthetic envelope validates only saved request records, never completion or identity.
    validate_records(dict(schema=1,family='whisper',engine='managed',conformance=True,held_outputs_unchanged=True,affinity=4,setup_seconds=0,flags={},records=records),prefix,'conformance')
    assert all(r['gc_after'][2]==20 for r in records)
    assert not (base/'campaign/timing').exists() and not (base/'campaign/conformance-gate.json').exists()
    terminal=ssh('''import sys
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
for b in %r:
 try:assert psutil.Process(b['pid']).create_time()!=b['birth']
 except psutil.NoSuchProcess:pass
print('terminal')
'''%receipt['births']);assert terminal.strip()=='terminal'
    output=dict(campaign_passed=False,closure_passed=True,reason='available-memory reserve',completed=observations,
                partial_calls=16,planned_managed_whisper_calls=20,partial_records=records,last_sample=last,
                failed_worker_samples=len(samples),peak_rss=run['peak_rss'],limits=LIMITS,births=receipt['births'],frozen=pin(base/'frozen.json'),timing_calls=0)
    folder=__import__('pathlib').Path(__file__).resolve().parent
    data=folder/'resource-failure-observations-20260920.json';write(data,output)
    report=folder/'resource-failure-20260920.md'
    text=f'''# AMD audio comparison stopped during Whisper conformance

The all-family campaign failed its unchanged available-memory guard before any
timing process began. Managed Whisper completed 16 of 20 requests; all sixteen
saved token/text/stop and input/ownership checks pass. This prefix does not qualify
the complete corpus, final held-output check or repeated-process memory behavior.

The last sample has **{last['available']:,} bytes available**, below the
**{LIMITS['available']:,}-byte reserve**, with **{run['peak_rss']:,} bytes RSS**.
The separate RSS, disk, time and CPU-affinity guards still pass. All preceding
978 samples pass every guard. The supervisor terminated its owned managed worker;
all seven original process identities are absent. The complete 979-sample sequence,
partial outputs and traceback remain intact. This is a resource failure, not a
Microsoft ORT timing comparison or a numerical mismatch.

Parakeet and pyannote both complete conformance under both engines: 48 calls in
four workers. Native Whisper also completes all twenty calls and the complete
Linux feature-array checks. Every complete worker's identity, request, output,
resource sample and original raw record was independently checked. No worker was
restarted, no tolerance changed and no timing number is inferred from conformance.

All sixteen saved managed Whisper calls allocate approximately 597–718 MB each.
After the first call, the generation-2 collection count stays at 20 through the
last saved call; younger-generation collections continue. These observations
support investigating allocation and collection behavior. They do not establish
which objects remain reachable or prove a leak. Neither forced collection nor a
larger memory limit is applied to turn this failed run into a success.

Hardware is AMD EPYC 9V74, CPU 2, .NET 10.0.8 / ORT 1.29.0, product source
`1d10d22`. Exact private and native-library identities remain in the original
frozen manifest, SHA-256 `{pin(base/'frozen.json')['sha256']}`.
The collection binds {len(receipt['files'])} files and verifies
{receipt['external_verified']:,} external model/library files. Archive SHA-256:
`{read(BASE/'collection-transfer.json')['archive']['sha256']}`.

A separate prospective Parakeet/pyannote timing lane may reuse their four complete
checks after verifying unchanged model, consumer, runtime and product identities.
It must retain this all-family failure and describe Whisper's AMD timing as pending.
The historical Windows comparison remains valid within its stated scope.
'''
    with report.open('x',encoding='utf-8') as f:f.write(text)
    write(BASE/'failure-audit.json',output)
    files={str(p.relative_to(ROOT)).replace('\\','/'):pin(p) for p in [BASE/'collection-transfer.json',base/'collection.json',BASE/'failure-audit.json',data,report,__import__('pathlib').Path(__file__).resolve()]}
    write(BASE/'failure-closed.json',dict(closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),campaign_passed=False,closure_passed=True,files=files,births=receipt['births']))
    print(json.dumps(dict(closed=True,campaign_passed=False,timing_calls=0,completed_conformance_calls=68,partial_calls=16,receipt=pin(BASE/'failure-closed.json'))))


if __name__=='__main__':main()
