"""Publish every frozen screen clock and independently rescore the campaign."""
from pathlib import Path
import csv,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/pyannote/winograd-screen'))
from score import score,ORDER
BASE=ROOT/'artifacts/pyannote-winograd-screen-amd-20260923'
OUT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf8'))
def pin(p):return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def table(path,rows):
    assert not path.exists()
    with path.open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    proof=read(BASE/'closed.json');assert proof['passed']
    for name,wanted in proof['files'].items():assert pin(BASE/name)==wanted,name
    analysis=read(BASE/'analysis.json');reports={name:read(BASE/'collected'/name/'result.json') for name in ORDER}
    performance=score(reports,read(BASE/'bundle/evidence/fixtures.json'),read(BASE/'bundle/reference.json'))
    assert performance==analysis['performance']
    table(OUT/'screen-clocks-20260923.csv',[dict(process=name,**r) for name in ORDER for r in reports[name]['observations']])
    table(OUT/'screen-preparation-20260923.csv',[dict(process=name,**r) for name in ORDER for r in reports[name]['preparation']])
    (OUT/'screen-observations-20260923.json').write_text(json.dumps(analysis,indent=2)+'\n',encoding='utf8')
    aggregate=performance['rows'][0];ratio=aggregate['ratio']['seconds'];gain=100*(1-ratio)
    lines=['# Complete-call Winograd screen','',
        '**'+('Admitted for separate product qualification.' if performance['admitted'] else 'Not admitted.')+'**','',
        f"The sum of all 87 eligible call means is {aggregate['current']['seconds']:.9f} seconds",
        f"for the selected direct helper and {aggregate['candidate']['seconds']:.9f} seconds for Winograd.",
        f"Candidate/current is **{ratio:.9f}**: **{abs(gain):.2f}% {'lower' if gain>=0 else 'higher'} component latency**.",
        'This is a standalone component screen; no product source is integrated and',
        'no new application or Microsoft ORT speed result is supplied.','',
        '| Form | Direct ms | Winograd ms | Candidate / current |','|---:|---:|---:|---:|']
    for r in performance['rows'][1:]:lines.append(f"| {r['form']} | {r['current']['seconds']*1000:.6f} | {r['candidate']['seconds']*1000:.6f} | {r['ratio']['seconds']:.6f} |")
    lines += ['', 'Four fresh processes run current, candidate, candidate, current. Every one',
        'of 4,176 call clocks is retained: 1,044 warmup and 3,132 measured, plus',
        '464 separate preparation clocks. Each captured call has three iterations',
        'per pass, fixed by geometry before timing. All eight forms retain their',
        'original multiplicities over three crops. No trimming, calibration or retry.','',
        'Exact-clock gates require at least 10% aggregate improvement, no form more',
        'than 5% slower, every role/process control within 1.10 aggregate and 1.20',
        'per form, and strict separation of candidate and current process means.','',
        f"Repeatability controls: {sum(r['passed'] for r in performance['controls'])}/18 pass.",
        f"Speed gates: {sum(r['passed'] for r in performance['gates'])}/9 pass.",
        f"Strict process separation: {'PASS' if performance['process_separation']['passed'] else 'FAIL'}.",'',
        '| Process | Sum of call means seconds | Preparation seconds | Retained prepared bytes |',
        '|---|---:|---:|---:|']
    for name in ORDER:
        lines.append(f"| {name} | {performance['process_totals'][name]['total']['seconds']:.9f} | {performance['preparation'][name]['seconds']:.9f} | {performance['prepared_bytes'][name]} |")
    failures=[r for r in performance['controls']+performance['gates'] if not r['passed']]
    if failures:
        lines += ['','Failed criteria:','']
        for r in failures:lines.append(f"- {r.get('role','speed')}, form {r['form']}: {r['ratio']['seconds']:.9f}, limit {r['limit']['seconds']:.9f}.")
    lines += ['', 'Timers include allocation of the owned result, scratch planning/rentals/',
        'returns, validation, finite checks, runtime transforms, convolution,',
        'conversions and bias/residual/ReLU. Loading, preparation, hashing and',
        'journal serialization are outside the call timer. Both roles call the',
        'same qualified binary through typed delegates. Every output matches its',
        'own qualified numerical hash; input and held-output ownership checks pass.',
        'Graph scheduling and the audio application are outside this boundary.',
        'The direct helper combines epilogues that the product sometimes schedules',
        'separately, so application gain cannot be inferred from this screen.','',
        'The new consumer is normally built before a separate 174-call verification',
        'worker checks both roles. Only afterward do the scored processes begin.',
        f"All {sum(r['samples'] for r in analysis['resources'])} resource observations pass; peak owned RSS is",
        f"{max(r['peak_rss'] for r in analysis['resources']):,} bytes. Every recorded owner is terminal.",
        'AMD EPYC 9V74 CPU2, SDK10.0.204/runtime10.0.8, ordinary AVX512 execution.',
        'No profiler, ISA or tiering override is used.','',
        'Qualified arithmetic binary:', '`937ab50e8d9cc140d7c27ba9d010bd8fe32638bc115173e436ef44a9646edd84`.',
        'Independent closure:',f"`{pin(BASE/'closed.json')['sha256']}`.",'',
        '[Every call clock](screen-clocks-20260923.csv),',
        '[every preparation clock](screen-preparation-20260923.csv),',
        '[all gates and resources](screen-observations-20260923.json),',
        '[frozen protocol](../winograd-screen/README.md).','',
        ('Next: isolated product dispatch and full suites, package, native/model, long-meeting and application qualification. The full 3% application gate still determines integration.' if performance['admitted'] else 'No product integration or unchanged timing retry follows this rejected screen.'),'']
    (OUT/'screen-20260923.md').write_text('\n'.join(lines),encoding='utf8',newline='\n')
    print(json.dumps(dict(admitted=performance['admitted'],ratio=ratio,report=pin(OUT/'screen-20260923.md'))))

if __name__=='__main__':main()
