"""Recompute integer timing totals and decisions, then close all evidence and report."""
from pathlib import Path
import argparse,datetime,json,math
from common import BANKS,CASES,VARIANTS,pin,read,write,verify
from vm import LOCAL_ORIGIN,ssh,prefix

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--artifact',type=Path,required=True);args=parser.parse_args();base=args.artifact.resolve();payload=base/'collected'
    assert not (base/'closed.json').exists();verify(payload,LOCAL_ORIGIN);audit=read(base/'audit.json');collection=read(base/'collection-check.json')
    assert audit['execution_passed'] and audit['bundle']==pin(payload/'bundle.json') and collection['passed'] and collection['archive']==pin(base/'results.tar.gz')
    expected=collection['remote']['collection']['files']|{'collection.json':collection['remote']['receipt']}
    assert {p.relative_to(payload).as_posix() for p in payload.rglob('*') if p.is_file()}==set(expected)
    for name,want in expected.items():assert pin(payload/name)==want,name
    # Separate computation uses raw integer sums, not the analyzer's row means.
    totals={};counts={'measured':0,'warmup':0,'first':0}
    for visit in range(4):
        timing=read(payload/'result'/str(visit)/'output/timing.json')
        for case in timing['results']:
            for label in counts:counts[label]+=len(case[label])
            for role in VARIANTS:
                records=[r for r in case['measured'] if r['variant']==role];assert len(records)==48
                ticks=sum(r['ticks'] for r in records);banks=sum(r['repeats'] for r in records);mean=ticks*1000/(timing['frequency']*banks)
                totals[(case['name'],visit,role)]=mean
                row=next(r for r in audit['rows'] if (r['case'],r['visit'],r['variant'])==(case['name'],visit,role))
                assert row['ticks']==ticks and row['banks']==banks and math.isclose(row['mean_ms'],mean,rel_tol=1e-12,abs_tol=1e-12)
    assert counts==dict(measured=6912,warmup=2304,first=144)
    controls=[];candidates=[]
    for definition,case in zip(BANKS,audit['cases']):
        name=definition['name'];assert case['name']==name
        means={role:sum(totals[(name,visit,role)] for visit in range(4))/4 for role in VARIANTS}
        for role in VARIANTS:assert math.isclose(means[role],case['means_ms'][role],rel_tol=1e-12,abs_tol=1e-12)
        control=1/1.01<=means['CopyB']/means['CopyA']<=1.01
        regression=all(means['Wide']/means[role]<=1.01 for role in VARIANTS[:3])
        gain=name not in CASES[1:4] or all(means['Wide']/means[role]<=.98 for role in VARIANTS[:3])
        for visit in range(4):
            control &= 1/1.02<=totals[name,visit,'CopyB']/totals[name,visit,'CopyA']<=1.02
            regression &= all(totals[name,visit,'Wide']/totals[name,visit,role]<=1.02 for role in VARIANTS[:3])
        assert (control,regression,gain)==(case['controls_passed'],case['no_regression_passed'],case['gain_passed'])
        controls.append(control);candidates.append(regression and gain)
    assert (all(controls),all(candidates),all(controls) and all(candidates))==(audit['controls_passed'],audit['candidate_screen_passed'],audit['overall_passed'])
    assert {(p['pid'],p['birth']) for p in audit['births']}=={(p['pid'],p['birth']) for p in collection['remote']['collection']['births']}
    terminal=json.loads(ssh(prefix()+'births=%r\n'%audit['births']+'''
for item in births:
 try:assert psutil.Process(item['pid']).create_time()!=item['birth'],item
 except psutil.NoSuchProcess:pass
print(json.dumps(dict(checked_at=time.time(),births=births)))
'''))
    write(base/'independent-verification.json',dict(passed=True,counts=counts,rows=len(totals),terminal=terminal,audit=pin(base/'audit.json')))
    report=Path(__file__).with_name('results-20260920.md');observations=Path(__file__).with_name('observations-20260920.json');assert not report.exists() and not observations.exists()
    write(observations,{k:v for k,v in audit.items() if k!='births'})
    lines=['# Complete LayerNorm-bank comparison — September 20, 2026','',f"**{audit['verdict']}**. Every measured output, held result, input, parameter and resource check passes. No production change or complete-model speedup follows from this component experiment.",'',
      'Four fresh sequential workers each execute all 25 captured LayerNorm calls for five e5 cases, plus four explicitly diagnostic thirty-row banks. The diagnostic banks map column j to j modulo 384 at widths 383 and 385, with and without bias. All statistics and output writes are inside timing; loading, hashing and independent validation are outside.','',
      '| Bank | Product ms | Copy A ms | Copy B ms | Wide ms | Wide / Product | Wide / Copy A | Wide / Copy B | Controls | Gain / regression screen |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---|---|']
    for c in audit['cases']:
        m=c['means_ms'];r=c['candidate_ratios'];lines.append(f"| {c['name']} | {m['Product']:.6f} | {m['CopyA']:.6f} | {m['CopyB']:.6f} | {m['Wide']:.6f} | {r['Product']:.6f} | {r['CopyA']:.6f} | {r['CopyB']:.6f} | {'pass' if c['controls_passed'] else 'fail'} | {'pass' if c['gain_passed'] and c['no_regression_passed'] else 'fail'} |")
    lines += ['', 'Means are milliseconds per complete 25-node bank, averaging four equal-sized visits. Copy A and Copy B call the same unchanged source-copy delegate; Product invokes the actual archived kernel. Every variant uses the same caller-owned input/output buffers. See [complete observations](observations-20260920.json) for every visit, distribution, allocation, GC count and resource summary.','',
      'All 6,912 measured batches, 2,304 fixed warmup batches and 144 first calls are retained. Each bank has 16 warmup and 48 measured cycles per worker, with four rotating variant positions. Batches contain 128/32/8/8/2 banks for the five real cases and 32 for each diagnostic. There is no convergence selection, discarded tail, forced GC, profiler or runtime override.','',
      'The prospective screen requires duplicate means within 1% overall and 2% per worker; at least 2% candidate improvement against all three controls at 30/padded128/128; no regression above 1% overall or 2% per worker on any bank. These empirical screens are not confidence intervals and do not establish independent per-call samples.','',
      'All real outputs match the original closed captures bit for bit. Diagnostic candidates/copies match actual product outputs exactly; all sixteen saved diagnostic arrays also pass independent centered-double scalar normalization at 1e-5. An actual first output remains held throughout subsequent calls. Every complete bank output hash matches after every batch; input and parameter hashes remain unchanged. The [prior AMD arithmetic/code proof](../layernorm-amd-proof/results-20260920.md) retains the broader exceptional-value and storage scope.','',
      'Linux .NET 10.0.8 on AMD EPYC 9V74, CPU 2 before CLR startup, supervisor CPU 0. Vector<float> uses eight lanes; AVX-512 is available. Core SHA256 is `'+CORE+'`. Kernel source is byte-identical to the closed hardware proof. This experiment neither invokes native ORT nor measures its latency.','',
      'All original PID/creation-time identities are terminal. Each worker remains within 600 seconds, 3 GiB sampled group RSS and 2 GiB available memory. Observed foreign CPU stays below 2% of machine capacity and guest steal below 0.5%; snapshots cannot observe all exited work or hypervisor interference.','',
      '| Visit | Seconds | Peak sampled RSS bytes | Minimum available bytes | Foreign CPU fraction | Steal fraction |',
      '|---|---:|---:|---:|---:|---:|']
    for r in audit['resources']:lines.append(f"| {r['visit']} | {r['seconds']:.3f} | {r['peak_rss']:,} | {r['minimum_available']:,} | {r['foreign']['foreign_cpu_fraction']:.6g} | {r['steal_fraction']:.6g} |")
    lines+=['',f"Frozen source `{read(payload/'bundle.json')['source']}`; bundle SHA256 `{pin(payload/'bundle.json')['sha256']}`; audit SHA256 `{pin(base/'audit.json')['sha256']}`.",
      'All sources, samples, diagnostic arrays, complete capture bindings and receipts are retained under `artifacts/e5-layernorm-bank-20260920`. Collection streams locally and does not duplicate the existing remote capture bank.','']
    with report.open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines))
    files={p.relative_to(base).as_posix():pin(p) for p in sorted(base.rglob('*')) if p.is_file()}
    write(base/'closed.json',dict(schema=1,execution_passed=True,performance_passed=audit['overall_passed'],closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,
        reports={p.as_posix():pin(p) for p in [report,observations]}))
    print('Closed',len(files),'files;',pin(base/'closed.json'))

if __name__=='__main__':
    from common import CORE
    main()
