"""Close full deployment evidence and permit comparison only after all matching controls pass."""
from pathlib import Path
import argparse,json,shutil,sys,time
import numpy as np
import audit as a
from vm import ssh,PSUTIL

ROOT=Path(__file__).resolve().parents[3]

def terminal(births):
    script='import sys,json,time\nsys.path.insert(0,%r)\nimport psutil\nrows=json.loads(%r)\n' % (PSUTIL,json.dumps(births))+'''
for row in rows:
 try:assert psutil.Process(row['pid']).create_time()!=row['birth'],row
 except psutil.NoSuchProcess:pass
print(json.dumps(dict(checked_at=time.time(),births=rows)))
'''
    return json.loads(ssh(script))

def report(result):
    lines=[]
    for case in result['timing']['cases']:
        for boundary,b in case['boundaries'].items():
            m=b['role_mean_seconds'];failed=[c['numerator']+'/'+c['denominator']+':'+k for c in b['controls'] for k,ok in c['gates'].items() if not ok]
            failed+=['candidate:'+k for k,ok in b['candidate_gates'].items() if not ok]
            lines.append(f"| {case['name']} | {case['policy']} | {boundary} | {m['A']*1000:.4f} | {m['B']*1000:.4f} | {m['C']*1000:.4f} | {m['N']*1000:.4f} | {b['candidate_ratio']:.6f} | {b['native_ratios']['C']:.6f} | {', '.join(failed) or 'pass'} |")
    return '\n'.join(lines)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--phase',choices=['aa','compare'],required=True);args=p.parse_args()
    base=args.artifact.resolve();phase=args.phase;payload=base/('collected-'+phase);source=Path(__file__).resolve().parent
    assert not (base/(phase+'-closed.json')).exists()
    collection=a.read(payload/('collection-'+phase+'.json'));check=a.read(base/('collection-check-'+phase+'.json'));meta=a.read(payload/'frozen.json')
    assert check['passed'] is True and check['streamed'] is True and check['remote']['collection']==collection
    assert check['archive']==a.pin(base/(phase+'-results.tar.gz'))
    assert a.pin(payload/('collection-'+phase+'.json'))==check['remote']['receipt']
    assert {p.relative_to(payload).as_posix() for p in payload.rglob('*') if p.is_file()}==set(collection['files'])|{'collection-'+phase+'.json'}
    for name,want in collection['files'].items():assert a.pin(payload/name)==want,name
    assert a.pin(payload/'frozen.json')==a.pin(base/'payload/frozen.json')==collection['frozen']
    result=a.audit(payload,phase);assert result==a.read(base/(phase+'-audit.json'))
    assert {(b['pid'],b['birth']) for b in collection['births']}=={(b['pid'],b['birth']) for b in result['resources']['births']}
    verified=terminal(collection['births'])
    sys.path.append(str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'));import psutil
    smoke=a.read(base/'smoke-audit.json')
    for b in smoke['resources']['births']:
        try:assert psutil.Process(b['pid']).create_time()!=b['birth'],b
        except psutil.NoSuchProcess:pass
    for relative,key in [('artifacts/e5-fingerprint-product-v2-20260920/closed.json','qualified_product_receipt'),('artifacts/e5-fingerprint-balanced-20260920/closed.json','common_state_receipt')]:
        assert a.pin(ROOT/relative)==meta[key]
    if phase=='compare':
        gate=a.read(base/'aa-gate.json');state=a.read(payload/'result-compare/identity.json')
        assert state['gate_sha256']==a.pin(base/'aa-gate.json')['sha256'] and gate['phase_receipt']==a.pin(base/'aa-closed.json')
        for name,want in gate['remote_files'].items():assert a.pin(payload/name)==want,name
    shutil.copytree(source,base/('closure-'+phase+'-source'),ignore=shutil.ignore_patterns('bin','obj','__pycache__','*results-20260920.md','*observations-20260920.json'))
    evidence={}
    for folder in ['payload','collected-'+phase,'smoke-process','bin','inputs','closure-'+phase+'-source']:
        for path in sorted((base/folder).rglob('*')):
            if path.is_file():evidence[path.relative_to(base).as_posix()]=a.pin(path)
    for name in ['smoke-audit.json','analysis-tests-final.log','build.log','preparation.json','deployment-'+phase+'.json','collection-check-'+phase+'.json',phase+'-results.tar.gz',phase+'-audit.json']:
        evidence[name]=a.pin(base/name)
    receipt=dict(schema=1,phase=phase,closed_at=time.time(),evidence_passed=True,timing_passed=result['timing']['passed'],audit=a.pin(base/(phase+'-audit.json')),
                 frozen=a.pin(payload/'frozen.json'),terminal=verified,evidence_files=evidence)
    a.write(base/(phase+'-closed.json'),receipt)
    if phase=='aa' and result['timing']['passed']:
        a.write(base/'aa-gate.json',dict(passed=True,timing_passed=True,frozen=a.pin(payload/'frozen.json'),identity=a.pin(payload/'result-aa/identity.json'),
            phase_receipt=a.pin(base/'aa-closed.json'),audit=a.pin(base/'aa-audit.json'),births=collection['births'],remote_files=collection['files']))
    prefix='aa' if phase=='aa' else 'comparison';observations=dict(audit=result,receipt=a.pin(base/(phase+'-closed.json')),verification=receipt)
    a.write(source/(prefix+'-observations-20260920.json'),observations)
    text=f'''# Independent e5 deployment: {prefix} — September 20, 2026

The fixed160-worker phase **{'passes' if result['timing']['passed'] else 'fails'} its prospective timing screen**.
All complete output, configuration, cache, ownership, identity and resource
checks pass. Every {result['measured_calls']:,} measured and {result['conditioning_calls']:,}
conditioning call is retained. Fresh native ORT timings use the same CPU and
inputs in separate sequential processes. No default changes in this report.

| Case | Policy | Boundary | A ms | B ms | C ms | ORT ms | C / mean(A,B) | C / ORT | Failed screens |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
{report(result)}

A/B disable the cache. C also disables it in A/A, and enables it only in the
conditional comparison. Thus an A/A C/control ratio is an identical-setting
comparison, not a cache gain. Disabled processes never construct the cache.
Every process owns its graph, weights and lifetime state; no inference workers
overlap. Default and Memory are separately measured public lifetime policies.
See the [fixed protocol](README.md) and [complete observations]({prefix}-observations-20260920.json)
for every visit, position, distribution, allocation and GC tail.

Both boundaries come from each individual call: public Execute/Run and the
enclosing Reset/disposal-plus-execution. Loading, first call and all thirty-second
conditioning observations are retained separately. Output copying and external
validation are outside timing. No forced GC, profiler, tiering override,
convergence selection or discarded observations are used. These empirical
screens are not confidence intervals or a claim of independent per-call samples.

Product is archive-qualified faf2844, core `{a.CORE}`; source-pinned probe and
runner revision `{meta['source_revision']}`. AMD EPYC9V74 uses.NET10.0.8/AVX512,
CPU2 inherited before CLR startup, supervisorCPU0. Native ORT1.23.2 uses one
intra/inter-op thread, sequential scheduling, all graph optimizations and no
spinning. Actual native SHA256 is `{a.NATIVE}`. Managed workers load no native ORT.

Every complete before/after array passes the unchanged1e-4 native scaled-error
gate; maximum observed error {result['maximum_native_error']:.12g}. Managed
output bytes match across all settings and visits. Actual first outputs and
inputs remain unchanged, as do prepared graph fingerprints and enabled cache
contents. All160workers and supervisor are terminal by original PID/birth.
All {sum(w['samples'] for w in result['resources']['workers']):,} resource samples remain. Bounds are300seconds,
6GiB group RSS,1GiB available, observed foreign CPU<=2%ofmachinecapacity and
gueststeal<=.5%. Snapshots miss some short-lived work and hypervisor neighbors.

Independent phase receipt `{observations['receipt']['sha256']}` binds retained
evidence under artifacts/{base.name}. Collection streams the archive locally,
without a second large VM copy. Successful workers and writers must not rerun.
'''
    if phase=='aa':text+='\nThe passing receipt permits the already frozen comparison phase.\n' if result['timing']['passed'] else '\nControls failed; the candidate phase is not run and the switch remains off.\n'
    else:text+='\nDefault promotion requires a passing comparison screen and subsequent current-source integration qualification.\n'
    with (source/(prefix+'-results-20260920.md')).open('x',encoding='utf-8') as f:f.write(text)
    print(json.dumps(dict(phase=phase,evidence_passed=True,timing_passed=result['timing']['passed'],receipt=observations['receipt'])))

if __name__=='__main__':main()
