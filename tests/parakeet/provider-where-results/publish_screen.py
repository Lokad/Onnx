"""Publish the rejected M57 screen with all clocks and descriptive phase blocks."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'artifacts/parakeet-provider-where-screen-amd-20260924'
OUT = Path(__file__).resolve().parent


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def main():
    assert pin(BASE/'closed.json')['sha256'] == 'a6fed3ecd03b7891a263da22d21645e172e81e62385542228a898dae1b049036'
    proof = json.loads((BASE/'closed.json').read_text())
    assert proof['passed'] and not proof['performance_admitted']
    for name,wanted in proof['files'].items():
        assert pin(BASE/name) == wanted, name
    analysis = json.loads((BASE/'analysis.json').read_text()); score = analysis['comparison']
    controls = [c for c in score['controls'] if not c['passed']]
    failed = [r for r in score['rows'] if not r['passed']]
    assert len(score['controls']) == 252 and len(controls) == 18 and len(failed) == 14
    assert sum(c['role']=='current' for c in controls)==7
    for source,target in [('clocks.csv','screen-clocks-20260924.csv'),('setups.json','screen-setups-20260924.json')]:
        assert not (OUT/target).exists(); shutil.copy2(BASE/source,OUT/target)
    report = OUT/'screen-observations-20260924.json'; assert not report.exists()
    report.write_text(json.dumps(dict(closure=pin(BASE/'closed.json'), analysis=analysis),indent=2)+'\n')
    cases = OUT/'screen-cases-20260924.csv'; assert not cases.exists()
    with cases.open('w',newline='',encoding='utf8') as stream:
        writer=csv.writer(stream); writer.writerow(['index','name','partition','current_seconds','candidate_seconds','candidate_current','case_gate'])
        for row in score['rows']:
            writer.writerow([row['index'],row['name'],row['partition'],row['current']['value'],row['candidate']['value'],row['ratio']['value'],row['passed']])
    # Every case and every sample appears in these descriptive fixed blocks.
    # This analysis changes neither weighting nor the original admission verdict.
    phase = OUT/'screen-phase-blocks-20260924.csv'; assert not phase.exists()
    with phase.open('w',newline='',encoding='utf8') as stream:
        writer=csv.writer(stream); writer.writerow(['process','case','block','warmup','first_iteration','samples','ticks','frequency','batch','mean_seconds'])
        for process in score['processes']:
            result=json.loads((BASE/'collected'/process/'result.json').read_text())
            for row in result['rows']:
                for block in range(6):
                    clocks=row['clocks'][block*20:(block+1)*20]
                    assert len(clocks)==20 and all(c['warmup']==(block<3) for c in clocks)
                    ticks=sum(c['ticks'] for c in clocks)
                    writer.writerow([process,row['name'],block,block<3,block*20,20,ticks,result['frequency'],row['batch'],ticks/(20*result['frequency']*row['batch'])])
    lines=['# Provider Where: complete-call performance comparison','',
        '**Candidate rejected by the frozen screen.** All correctness and resource checks pass, but 18 of 252 repeatability controls and 14 of 122 per-case regression gates fail. No integration or application-speed claim follows.','',
        '| Sum of per-case means | Selected ms | Candidate ms | Candidate / selected |',
        '|---|---:|---:|---:|']
    for key,label in [('target6','Six captured Parakeet uniform cases'),('other_uniform42','42 other uniform cases'),('fallback74','74 remaining cases'),('all122','All 122 cases')]:
        v=score['scopes'][key]; lines.append(f"| {label} | {v['current']['value']*1000:.6f} | {v['candidate']['value']*1000:.6f} | {v['ratio']['value']:.6f} |")
    lines += ['',
        'The target improvement and strict separation gates pass, as do all eight aggregate repeatability checks. Individual repeatability fails for seven selected and eleven candidate cases. The two large mixed-mask cases capture-5-last and capture-7-last regress by 13.998% and 7.546%; the smaller outer-mask-last case is 5.334x slower. These failures remain part of the verdict.','',
        'The retained clocks also show large changes within the unchanged selected release: vector-last differs by 8.158x between its fresh processes. Its first process has measured block means near 3.0 us; the last process is near 0.36 us. Outer-mask-first changes from approximately 20–32 us during warmup to approximately 4 us during measurement in the last selected process. This motivates investigating promotion and generated code; it does not prove a JIT or GC cause. All six fixed 20-sample blocks for every case/process are published without dropping or reweighting clocks.','',
        'Four fresh CPU2 processes run selected/candidate/candidate/selected on .NET 10.0.8 with ordinary settings and no profiler. All 122 prospectively fixed valid cases are included, with 60 warmup and 60 measured samples and deterministic batching. The timer surrounds complete CPUExecutionProvider.Where calls and assignment of owned OpResult values to a preallocated array. Metadata, exact values/shapes, all input stores, output ownership and held-output checks occur outside timing.','',
        f"All seven jobs, {analysis['resources']} resource samples, 488 setups and {score['sample_clocks']:,} clocks pass. There are {score['measured_clocks']:,} measured clocks, {score['public_calls']:,} complete public calls and {score['measured_public_calls']:,} measured public calls. Peak RSS is {analysis['peak_rss']:,} bytes. Owner893235/birth1790214291.23 and all descendants are terminal, code0.",'',
        f"Selected Core672e5f30/Data065b7a7f; candidate Cored8a8d8eb/Dataec483e1a. Consumer SHA-256:{analysis['consumer']['sha256']}. Frozen tools:d49e98d8; archivee5a4ab28,stage51fa4538,payload5d7e893e. Closure:a6fed3ecd03b7891a263da22d21645e172e81e62385542228a898dae1b049036.",'',
        '[Every case](screen-cases-20260924.csv), [every clock](screen-clocks-20260924.csv), [all setups](screen-setups-20260924.json), [exact scoring and identities](screen-observations-20260924.json), and [all phase blocks](screen-phase-blocks-20260924.csv) are retained. Complete local evidence is artifacts/parakeet-provider-where-screen-amd-20260924. Root product and BENCHMARK.md remain the fully qualified selected release.','']
    target=OUT/'screen-20260924.md'; assert not target.exists(); target.write_text('\n'.join(lines))
    print(json.dumps(dict(report=pin(target),failed_controls=len(controls),failed_cases=len(failed))))


if __name__=='__main__':
    main()
