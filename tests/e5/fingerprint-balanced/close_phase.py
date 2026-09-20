"""Independently close each immutable phase; only passing A/A can create a comparison gate."""
from pathlib import Path
from unittest.mock import patch
import argparse,copy,json,math,shutil,subprocess,sys,time
import numpy as np
import audit as checks
from vm import ssh

ROOT=Path(__file__).resolve().parents[3]

def telemetry(state,samples):
    expected=[dict(name=f'v{v}-{checks.CASES[i]}',case=checks.CASES[i],case_index=i,visit=v)
              for v in range(4) for i in (range(5) if v%2==0 else reversed(range(5)))]
    assert [r['job'] for r in state['runs']]==expected
    assert state['limits']==dict(seconds=300,rss=6*1024**3,available=1024**3)
    assert state['complete'] is True and state['code']==0
    previous=state['started'];count=0
    for run in state['runs']:
        assert all(math.isfinite(run[k]) for k in ['started','ended','seconds'])
        assert previous<=run['started']<run['ended']<=state['ended'];previous=run['ended']
        assert run['code']==0 and 0<run['seconds']<300
        rows=samples[run['job']['name']];assert len(rows)==run['samples']>0
        seen={};last=-1.;peak=0
        for row in rows:
            assert math.isfinite(row['seconds']) and last<=row['seconds']<=run['seconds'];last=row['seconds']
            assert type(row['available']) is int and row['available']>=1024**3
            assert len({p['pid'] for p in row['members']})==len(row['members'])
            for p in row['members']:
                assert type(p['pid']) is int and p['pid']>0 and math.isfinite(p['birth']) and p['birth']>=run['child']['birth']
                assert type(p['rss']) is int and p['rss']>=0 and p['affinity']==[2]
                assert seen.get(str(p['pid']),p['birth'])==p['birth'];seen[str(p['pid'])]=p['birth']
            total=sum(p['rss'] for p in row['members']);assert total<6*1024**3;peak=max(peak,total)
        assert seen==run['members'] and seen[str(run['child']['pid'])]==run['child']['birth']
        assert peak==run['peak_rss'];count+=len(rows)
    return count

def terminal(births):
    script="""import sys,json,time
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
rows=json.loads(%r)
for row in rows:
 try:assert psutil.Process(row['pid']).create_time()!=row['birth'],row
 except psutil.NoSuchProcess:pass
print(json.dumps(dict(checked_at=time.time(),births=rows)))
""" % json.dumps(births)
    return json.loads(ssh(script))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--phase',choices=['aa','compare'],required=True);a=p.parse_args()
    base=a.artifact.resolve();phase=a.phase;payload=base/('collected-'+phase);source=Path(__file__).resolve().parent
    assert not (base/(phase+'-closed.json')).exists()
    meta=checks.read(payload/'frozen.json');collection=checks.read(payload/('collection-'+phase+'.json'));check=checks.read(base/('collection-check-'+phase+'.json'))
    assert check['passed'] is True and check['remote']['collection']==collection
    assert checks.pin(base/(phase+'-results.tar.gz'))==check['remote']['archive']
    assert checks.pin(payload/('collection-'+phase+'.json'))==check['remote']['receipt']
    assert {p.relative_to(payload).as_posix() for p in payload.rglob('*') if p.is_file()}==set(collection['files'])|{'collection-'+phase+'.json'}
    for name,wanted in collection['files'].items():assert checks.pin(payload/name)==wanted,name
    assert checks.pin(base/'payload/frozen.json')==checks.pin(payload/'frozen.json')==collection['frozen']
    for name,wanted in meta['files'].items():assert checks.pin(base/'payload'/name)==wanted,name
    result=checks.audit(payload,phase);assert result==checks.read(base/(phase+'-audit.json'))
    state=checks.read(payload/('result-'+phase)/'identity.json')
    samples={r['job']['name']:[json.loads(line) for line in (payload/('result-'+phase)/r['job']['name']/'samples.jsonl').read_text().splitlines()] for r in state['runs']}
    sample_count=telemetry(state,samples)
    first_name=state['runs'][0]['job']['name']
    first_observed=next(i for i,row in enumerate(samples[first_name]) if row['members'])
    resources_refused=[]
    mutations=[('sample-time',lambda s,r:r[s['runs'][0]['job']['name']][0].update(seconds=301)),
        ('sample-count',lambda s,r:s['runs'][0].update(samples=0)),('peak',lambda s,r:s['runs'][0].update(peak_rss=0)),
        ('available',lambda s,r:r[s['runs'][0]['job']['name']][0].update(available=0)),
        ('affinity',lambda s,r:r[first_name][first_observed]['members'][0].update(affinity=[0])),
        ('birth',lambda s,r:s['runs'][0]['child'].update(birth=0)),('schedule',lambda s,r:s['runs'].reverse()),
        ('duplicate-member',lambda s,r:r[first_name][first_observed]['members'].append(r[first_name][first_observed]['members'][0].copy()))]
    for name,change in mutations:
        damaged=copy.deepcopy(state);altered=copy.deepcopy(samples);change(damaged,altered)
        try:telemetry(damaged,altered)
        except AssertionError:resources_refused.append(name)
        else:raise AssertionError('Damaged resources accepted: '+name)
    assert {tuple(sorted(b.items())) for b in collection['births']}=={tuple(sorted(b.items())) for b in result['resources']['births']}
    actual_terminal=terminal(collection['births'])
    # Verify local functional proof births too; completed evidence is not restarted.
    sys.path.append(str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
    import psutil
    smoke=checks.read(base/'smoke-process/identity.json');local_births=[smoke['supervisor']]+[dict(pid=int(pid),birth=birth) for pid,birth in smoke['members'].items()]
    for item in local_births:
        try:assert psutil.Process(item['pid']).create_time()!=item['birth'],item
        except psutil.NoSuchProcess:pass
    assert smoke['complete'] is True and smoke['code']==0
    product=ROOT/'artifacts/e5-fingerprint-product-v2-20260920/closed.json'
    assert checks.pin(product)==meta['qualified_product_receipt']
    evidence={}
    for directory in ['payload','collected-'+phase,'smoke-process','inputs','bin']:
        for path in sorted((base/directory).rglob('*')):
            if path.is_file():evidence[path.relative_to(base).as_posix()]=checks.pin(path)
    for name in [phase+'-audit.json',phase+'-results.tar.gz','collection-check-'+phase+'.json','smoke-audit.json','analysis-tests.log','preparation.json','deployment-'+phase+'.json']:
        evidence[name]=checks.pin(base/name)
    folder=base/('closure-'+phase+'-source');shutil.copytree(source,folder,ignore=shutil.ignore_patterns('bin','obj','__pycache__','*results-20260920.md','*observations-20260920.json'))
    for path in folder.rglob('*'):
        if path.is_file():evidence[path.relative_to(base).as_posix()]=checks.pin(path)
    receipt=dict(schema=1,phase=phase,closed_at=time.time(),evidence_passed=True,timing_passed=result['timing']['passed'],frozen=checks.pin(payload/'frozen.json'),
        audit=checks.pin(base/(phase+'-audit.json')),collection=checks.pin(payload/('collection-'+phase+'.json')),terminal=actual_terminal,
        sample_count=sample_count,resource_refusals=resources_refused,local_births=local_births,evidence_files=evidence)
    checks.write(base/(phase+'-closed.json'),receipt)
    if phase=='aa' and result['timing']['passed']:
        gate=dict(passed=True,timing_passed=True,frozen=checks.pin(payload/'frozen.json'),identity=checks.pin(payload/'result-aa/identity.json'),
            phase_receipt=checks.pin(base/'aa-closed.json'),audit=checks.pin(base/'aa-audit.json'),births=collection['births'],remote_files=collection['files'])
        checks.write(base/'aa-gate.json',gate)
    if phase=='compare':
        gate=checks.read(base/'aa-gate.json');assert state['gate_sha256']==checks.pin(base/'aa-gate.json')['sha256']
        assert gate['phase_receipt']==checks.pin(base/'aa-closed.json') and gate['audit']==checks.pin(base/'aa-audit.json')
        for name,wanted in gate['remote_files'].items():assert checks.pin(payload/name)==wanted,name
    prefix='aa' if phase=='aa' else 'comparison'
    observations=dict(audit=result,receipt=checks.pin(base/(phase+'-closed.json')),verification=receipt)
    checks.write(source/(prefix+'-observations-20260920.json'),observations)
    lines=[]
    for item in result['timing']['cases']:
        for boundary,b in item['boundaries'].items():
            means=b['role_mean_seconds'];failed=[f"{c['numerator']}/{c['denominator']}:{key}" for c in b['controls'] for key,ok in c['gates'].items() if not ok]
            failed += ['candidate:'+key for key,ok in b['candidate_gates'].items() if not ok]
            lines.append(f"| {item['name']} | {boundary} | {means[0]*1000:.4f} | {means[1]*1000:.4f} | {means[2]*1000:.4f} | {b['candidate_ratio']:.6f} | {', '.join(failed) or 'pass'} |")
    verdict='passes' if result['timing']['passed'] else 'fails'
    text=f'''# Locally balanced single-graph e5: {prefix} — September 20, 2026

The fixed twenty-worker {prefix} phase **{verdict} its prospective timing screen**.
All complete output, ownership, cache-state, identity and resource checks pass.
Each worker uses eight locally balanced six-cycle blocks, with unchanged limits.
Every one of the {result['measured_calls']:,} measured calls and
{result['conditioning_calls']:,} conditioning calls is retained. No default or
Microsoft ORT scoreboard changes follow from this common-state diagnostic.

| Case | Boundary | A ms | B ms | C ms | C / mean(A,B) | Failed screens |
|---|---|---:|---:|---:|---:|---|
{chr(10).join(lines)}

All three labels disable the cache in A/A. In comparison, A/B disable it and C
enables it. Every role uses the same prepared graph, weights, release buffers
and immutable string cache. The property changes outside timing; the cache
remains resident even for disabled calls. See the [fixed protocol](README.md)
for counts, ordering and gates. These empirical screens are not confidence
intervals or evidence that calls are independent. The worker and ordering
contrasts, medians, maxima, GC tails and allocation are all retained in the
[complete observations]({prefix}-observations-20260920.json).

Actual core `{checks.CORE}` is the previously qualified archive build from
`faf2844`; the probe and deployed runner source is `{meta['source_revision']}`. AMD EPYC 9V74
uses .NET 10.0.8 with CPU2 inherited before CLR startup. No forced GC, profiling
or tiering override was used. All complete before/after outputs match bits and
native references at the unchanged scaled-error limit of 1e-4; maximum error
is {result['maximum_native_error']:.12g}. Inputs, held outputs, exact fingerprints
and prepared cache contents remain unchanged. No native ORT inference or timing
was performed.

All twenty workers and the supervisor are terminal, verified by original PID
and birth. Every {sample_count:,} resource sample is retained. Limits remain
300 seconds, 6 GiB group RSS, 1 GiB available, observed foreign CPU <=2% and
guest steal <=0.5%. Snapshots miss some exited activity and do not control
hypervisor neighbors. Eight damaged real resource records are refused; the
pre-run tests cover eighteen damaged real smoke records, timing biases,
ordering and retention of GC tails.

The independent phase receipt is
`{observations['receipt']['sha256']}`. Its evidence inventory is immutable under
`artifacts/{base.name}`. Completed workers and writers must not be rerun.
'''
    if phase=='aa':
        text += '\nA passing independently closed A/A gate permits the already frozen comparison phase.\n' if result['timing']['passed'] else '\nThe candidate phase is not run: the matching A/A requirement failed. The switch remains off.\n'
    else:text += '\nA passing common-state result still requires separately declared isolated deployment evidence before default promotion.\n'
    (source/(prefix+'-results-20260920.md')).write_text(text,encoding='utf-8')
    print(json.dumps(dict(phase=phase,evidence_passed=True,timing_passed=result['timing']['passed'],receipt=observations['receipt'])))

if __name__=='__main__':main()
