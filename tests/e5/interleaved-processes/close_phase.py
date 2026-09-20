"""Close complete evidence; only passing matching A/A can authorize comparison."""
from pathlib import Path
import argparse,json,sys,time
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

def report(result,meta):
    lines=[];bridges=[]
    for case in result['timing']['cases']:
        for boundary,b in case['boundaries'].items():
            m=b['role_mean_seconds']
            failed=[c['numerator']+'/'+c['denominator']+':'+k for c in b['controls'] for k,ok in c['gates'].items() if not ok]
            failed+=['candidate:'+k for k,ok in b['candidate_gates'].items() if not ok]
            if not b['bridges_passed']:failed.append('solo/resident')
            lines.append(f"| {case['case']} | {case['policy']} | {boundary} | {m['A']*1000:.4f} | {m['B']*1000:.4f} | {m['C']*1000:.4f} | {m['N']*1000:.4f} | {b['candidate_ratio']:.6f} | {b['native_ratios']['C']:.6f} | {', '.join(failed) or 'pass'} |")
            bridges.extend(b['bridges'])
    table='\n'.join(lines);phase=result['phase'];workers=result['resources']['cohorts']
    conclusion='The independently closed A/A receipt permits the already frozen comparison.' if phase=='aa' and result['timing']['passed'] else 'Controls failed; the candidate phase is not run.' if phase=='aa' else 'This bundle comparison does not separate the two mechanisms or change production defaults automatically.'
    return f'''# Interleaved independent e5 processes: {phase} — September 20, 2026

The complete forty-cohort / 160-worker phase **{'passes' if result['timing']['passed'] else 'fails'}
its fixed timing screens**. Complete numerical, configuration, ownership,
process and resource checks pass. All {result['measured_calls']:,} measured,
{result['conditioning_calls']:,} conditioning and {result['solo_calls']:,} solo
calls are retained. No production default changes in this report.

| Case | Policy | Boundary | A ms | B ms | C ms | ORT ms | C / mean(A,B) | C / ORT | Failed screens |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
{table}

A/B use current defaults. C is identical in A/A and enables fingerprint strings
and wider LayerNorm together only in comparison. An A/A C/control ratio is not
a gain. Each engine has its own process, runtime, collector, graph and weights.
Only the active process is resumed; other resident processes are suspended.
Physical caches and memory still interact. These are resident-process timings,
not isolated-deployment estimates. The preceding isolated-deployment failure
remains unchanged.

Each worker conditions until both 128 calls and thirty cumulative execution
seconds. All first calls and conditioning are retained. Forty-eight measured
cycles comprise two locally balanced blocks of all 24 role permutations.
Batches have32/16/4/4/2calls by length. The first-created process also measures
64solo calls before others load; the last measures64after the others actually
terminate. Every role occupies both positions once per case/policy. Resident/solo
ratios across both boundaries range {min(b['ratio'] for b in bridges):.6f} to
{max(b['ratio'] for b in bridges):.6f}; every bridge remains in the observations,
including failures. Solo order and elapsed time can also affect these contrasts.

Public Execute/Run and enclosing Reset/disposal-plus-execution are measured
on every call. Pipe traffic, copying, loading and external validation are outside
the boundaries. Timing storage is allocated before measurement. No forced GC,
profiler, runtime override, fitted correction or sample exclusion is used.
Four fresh cohorts do not establish independent per-call samples or confidence
bounds. Complete visit/position screens, distributions, allocation and GC tails
are retained in the machine-readable observations.

Qualified product source4f10e8b, core `{a.CORE}`. Probe/runner source
`{meta['source_revision']}` is a source-pinned local build against that archive.
AMD EPYC9V74 uses.NET10.0.8, AVX-512, CPU2 inherited before runtime startup;
the supervisor uses CPU0. Native ORT1.23.2 uses one thread, sequential execution,
all graph optimizations and no spinning; actual SHA256 `{a.NATIVE}`.
Managed workers load no native ORT. Actual startup switches are verified.

Every complete before/after output matches its retained first output, inputs
and held results stay unchanged, and all native scaled errors pass1e-4;
maximum observed error {result['maximum_native_error']:.12g}. Managed output bytes
match across settings and visits. Fingerprints and cache states remain valid.

All original process births are terminal. The {sum(w['samples'] for w in workers):,}
resource samples remain, under600seconds per cohort,12GiB combined RSS and1GiB
minimum available memory. Observed foreign CPU<=2%andgueststeal<=0.5%pass.
Inactive processes are observed stopped and their CPU counters remain stable.
Snapshots miss some short-lived work and cannot observe hypervisor neighbors.

See [README.md](README.md) for the frozen protocol, and the corresponding
`{phase}-observations-20260920.json` for all evidence. Artifacts are retained at
`artifacts/e5-interleaved-processes-v3-20260920`. Collection streams locally and
verifies every file. The phase receipt binds the complete evidence and reports
after writers finish; successful stages must not be rerun.

{conclusion}
'''

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--phase',choices=['aa','compare'],required=True);args=p.parse_args()
    base=args.artifact.resolve();phase=args.phase;payload=base/('collected-'+phase);source=Path(__file__).resolve().parent
    assert not (base/(phase+'-closed.json')).exists()
    collection=a.read(payload/('collection-'+phase+'.json'));check=a.read(base/('collection-check-'+phase+'.json'));meta=a.read(payload/'frozen.json')
    assert check['passed'] and check['streamed'] and check['remote']['collection']==collection
    assert check['archive']==a.pin(base/(phase+'-results.tar.gz')) and a.pin(payload/('collection-'+phase+'.json'))==check['remote']['receipt']
    assert {p.relative_to(payload).as_posix() for p in payload.rglob('*') if p.is_file()}==set(collection['files'])|{'collection-'+phase+'.json'}
    for name,wanted in collection['files'].items():assert a.pin(payload/name)==wanted,name
    assert a.pin(payload/'frozen.json')==a.pin(base/'payload/frozen.json')==collection['frozen']
    result=a.audit(payload,phase);assert result==a.read(base/(phase+'-audit.json'))
    assert {(b['pid'],b['birth']) for b in collection['births']}=={(b['pid'],b['birth']) for b in result['resources']['births']}
    verified=terminal(collection['births'])
    sys.path.append(str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'));import psutil
    smoke=a.read(base/'smoke-audit.json');local={(b['pid'],b['birth']) for v in smoke['phases'].values() for b in v['resources']['births']}
    for pid,birth in local:
        try:assert psutil.Process(pid).create_time()!=birth,(pid,birth)
        except psutil.NoSuchProcess:pass
    assert a.pin(ROOT/'artifacts/e5-layernorm-product-20260920/closed.json')==meta['qualified_product_receipt']
    assert a.pin(ROOT/'artifacts/e5-deployment-variance-20260920/closed.json')==meta['diagnostic_receipt']
    if phase=='compare':
        gate=a.read(base/'aa-gate.json');state=a.read(payload/'result-compare/identity.json')
        assert state['gate_sha256']==a.pin(base/'aa-gate.json')['sha256'] and gate['phase_receipt']==a.pin(base/'aa-closed.json')
        for name,wanted in gate['remote_files'].items():assert a.pin(payload/name)==wanted,name
    verification=dict(remote_terminal=verified,local_terminal=[dict(pid=p,birth=b) for p,b in sorted(local)])
    observations=source/(phase+'-observations-20260920.json');report_path=source/(phase+'-results-20260920.md')
    a.write(observations,dict(audit=result,verification=verification))
    with report_path.open('x',encoding='utf-8') as stream:stream.write(report(result,meta))
    files={p.relative_to(base).as_posix():a.pin(p) for p in sorted(base.rglob('*')) if p.is_file()}
    reports={p.relative_to(ROOT).as_posix():a.pin(p) for p in [observations,report_path,Path(__file__).resolve()]}
    receipt=dict(schema=1,phase=phase,closed_at=time.time(),evidence_passed=True,timing_passed=result['timing']['passed'],
        audit=a.pin(base/(phase+'-audit.json')),frozen=a.pin(payload/'frozen.json'),verification=verification,files=files,reports=reports)
    a.write(base/(phase+'-closed.json'),receipt)
    if phase=='aa' and result['timing']['passed']:
        a.write(base/'aa-gate.json',dict(passed=True,timing_passed=True,frozen=a.pin(payload/'frozen.json'),identity=a.pin(payload/'result-aa/identity.json'),
            phase_receipt=a.pin(base/'aa-closed.json'),audit=a.pin(base/'aa-audit.json'),births=collection['births'],remote_files=collection['files']))
    print(json.dumps(dict(phase=phase,evidence_passed=True,timing_passed=result['timing']['passed'],receipt=a.pin(base/(phase+'-closed.json')))))

if __name__=='__main__':main()
