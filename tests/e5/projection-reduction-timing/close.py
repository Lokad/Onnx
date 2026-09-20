"""Close one complete comparison, without changing frozen timing policy or data."""
import argparse, datetime
from common import *
from audit import audit
from vm import ssh, REMOTE


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);a=p.parse_args();base=a.artifact.resolve()
    result=audit(base);assert result==read(base/'audit.json')
    terminal=json.loads(ssh('import sys,json\nsys.path.insert(0,%r)\nimport process_support as h\nrows=%r\n' % (REMOTE,result['resources']['births'])+'''
out=[]
for row in rows:
 actual=h.proc(row['pid']);assert actual is None or actual['start']!=row['start'],row
 out.append(dict(expected=row,actual=actual,terminal=True))
print(json.dumps(out))
'''))
    performance=result['performance'];lane=Path(__file__).resolve().parent
    data=lane/'observations-20260920.json';report=lane/'results-20260920.md'
    write(data,dict(audit=result,terminal=terminal))
    lines=['# Complete projection-bank reduction-block comparison — September 20, 2026','',
        f"Fixed verdict: **{performance['verdict'].upper()}**. Integrity, complete output-hash checks, ownership, affinity, resource and measured main-thread allocation/GC checks pass. Duplicate-original timing controls "+('pass.' if performance['controls_passed'] else 'fail; no candidate can qualify.'),'',
        '| Bank geometry | Original A ms | Identical original B ms | Block128 ms | Block256 ms |',
        '|---|---:|---:|---:|---:|']
    for row in performance['banks']:lines.append('| '+row['name']+' | '+' | '.join(f'{value:.6f}' for value in row['means_ms'])+' |')
    lines+=['',
        'Means above average the four worker means. Each timed visit clears destinations and computes all72 independent prepared matrices in twelve-layer order, including delegate dispatch and original row tails. Weights occupy84,934,656 bytes. These are deterministic synthetic e5 projection geometries, not complete e5 inference. The padded128 bank differs by seed, not attention-mask behavior. Packing and verification are outside these steady-state times.','',
        '| Bank | Original B/A aggregate | Worker B/A range | Order contrast | Controls pass | Block128 screen | Block256 screen |',
        '|---|---:|---:|---:|---|---|---|']
    for row in performance['banks']:
        control=row['control'];ratios=control['workers']
        lines.append(f"| {row['name']} | {control['aggregate']:.6f} | {min(ratios):.6f}–{max(ratios):.6f} | {control['order_contrast']:.6f} | {control['passed']} | {row['candidates']['2']['screen_passed']} | {row['candidates']['3']['screen_passed']} |")
    lines+=['',
        'The prospective screen requires duplicate controls within2% aggregated,5% per worker and5% for the two ordering strata. Each candidate must improve EACH of30/padded128/128 by at least2% against BOTH originals, with primary worker ratios<=1.02 and aggregate8/512 ratios<=1.01. Aggregates use geometric means of worker ratios. Both candidates and every failure are retained; calls within a worker are not independent replicates and this is not a calibrated confidence interval.','']
    for mode,block in [(2,128),(3,256)]:
        lines.append(f"Block{block}: candidate gain/regression screen **{'passes' if performance['candidate_screens'][str(mode)] else 'fails'}**.")
    lines+=['', 'Nominated for a distinct product-integration proposal: '+({2:'block128',3:'block256'}.get(performance['nominated'],'none'))+'. No product integration, default promotion, full-model speedup or native ORT parity follows from this component run.','',
        f"All four fresh workers retain{result['counts']['measured']:,} measured,{result['counts']['conditioning']:,} conditioning and80 first-timed-bank calls. First calls follow construction of original verification hashes; they are not cold first-kernel invocation times. Each bank conditions for at least16complete cycles and3compute seconds in each mode. All48 measured cycles use the two frozen24-permutation blocks. No sample was dropped. Measured main-thread allocation and GC counts are zero; summed total-runtime allocation observations are{result['measured_total_allocated']:,}bytes.",'',
        'Every bank verifies all72 complete output hashes for all four modes before and after timing; inputs and packed weights are unchanged and all workers agree. These are assertions in the pinned consumer. Complete output arrays are not separately stored, and numerical outputs are not inspected during each timed visit. The earlier separately closed AMD proof supplies full original arrays and scalar-FMA checks.','',
        '| Worker | Elapsed seconds | Resource samples | Peak group RSS bytes |',
        '|---|---:|---:|---:|']
    for run in result['resources']['runs']:lines.append(f"| {run['name']} | {run['seconds']:.3f} | {run['samples']} | {run['peak_rss']} |")
    lines+=['',
        'Actual host is AMD EPYC9V74, normal .NET10.0.8, CPU2 inherited before runtime startup; supervisor CPU0. Fixed guards are600seconds/2GiB sampled process-group RSS/1GiB available memory. Guest CPU/steal counters and boundary foreign-process observations are retained. Boundary snapshots can miss short-lived activity and guest exclusivity does not establish hypervisor exclusivity. All five original process births are terminal.','',
        f"Frozen producer source `{result['source']}`. Reused proof Probe.dll `{PROBE}`, qualified core `{CORE}`. The new consumer is `{read(base/'payload/bundle.json')['files']['bin/Timing.dll']['sha256']}`. Its local build has zero errors and two CA1416 affinity warnings, retained in build logs; no kernel rebuild occurred. No source correction or policy change was made after observing timings.",'',
        '[Protocol and commands](README.md), [complete observations](observations-20260920.json), and [prior AMD proof](../projection-reduction-blocks/proof-results-20260920.md). All raw timings, supervisor samples, schedules, inputs-by-hash, preparation/build logs, archives and file manifests are retained in `artifacts/e5-reduction-timing-20260920`.','']
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
    receipt=dict(closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),integrity_passed=True,verdict=performance['verdict'],
        reports={p.relative_to(ROOT).as_posix():pin(p) for p in [data,report,Path(__file__)]},births=result['resources']['births'],
        files={p.relative_to(base).as_posix():pin(p) for p in sorted(base.rglob('*')) if p.is_file()})
    write(base/'closed.json',receipt);print(json.dumps(dict(receipt=pin(base/'closed.json'),files=len(receipt['files']),verdict=performance['verdict'])))


if __name__=='__main__':main()
