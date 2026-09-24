"""Publish all control cases and phase blocks; retain raw clocks in the closed archive."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'artifacts/parakeet-dense-scalar-where-stability-amd-20260924'
OUT = Path(__file__).resolve().parent


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def main():
    assert pin(BASE/'closed.json')['sha256']=='882ac1267e5ef80fc7b8bf2b01f3252f5b72cba8fb2fc1da2c783fa9e7e27986'
    proof = json.loads((BASE/'closed.json').read_text())
    assert proof['passed']
    for name,wanted in proof['files'].items(): assert pin(BASE/name)==wanted,name
    a=json.loads((BASE/'analysis.json').read_text());s=a['comparison']
    assert a['identical_binary_control'] and not a['candidate_measured']
    assert a['products']['current']==a['products']['candidate']
    assert a['stability_admitted']==proof['stability_admitted']==s['admitted']
    assert len(s['controls'])==225 and len(s['rows'])==220
    assert s['sample_clocks']==686400 and s['measured_clocks']==158400 and a['setup_count']==880
    failed_controls=[r for r in s['controls'] if not r['passed']]
    failed_cases=[r for r in s['rows'] if not r['passed']]
    cases=OUT/'stability-cases-20260924.csv';assert not cases.exists()
    with cases.open('w',newline='',encoding='utf8') as stream:
        w=csv.writer(stream);w.writerow(['index','name','partition','outer_seconds','middle_seconds','middle_outer','symmetric_five_percent_pass'])
        for r in s['rows']:w.writerow([r['index'],r['name'],r['partition'],r['outer']['value'],r['middle']['value'],r['ratio']['value'],r['passed']])
    blocks=OUT/'stability-phase-blocks-20260924.csv';assert not blocks.exists()
    with blocks.open('w',newline='',encoding='utf8') as stream:
        w=csv.writer(stream);w.writerow(['process','case','block','warmup','first_iteration','samples','ticks','frequency','batch','mean_seconds'])
        for process in s['processes']:
            value=json.loads((BASE/'collected'/process/'result.json').read_text())
            for row in value['rows']:
                for block in range(13):
                    clocks=row['clocks'][block*60:(block+1)*60]
                    assert len(clocks)==60 and all(c['warmup']==(block<10) for c in clocks)
                    ticks=sum(c['ticks'] for c in clocks)
                    w.writerow([process,row['name'],block,block<10,block*60,60,ticks,value['frequency'],row['batch'],ticks/(60*value['frequency']*row['batch'])])
    raw={name:dict(path=(BASE/name).relative_to(ROOT).as_posix(),**pin(BASE/name)) for name in ['clocks.csv','setups.json','results.tar.gz']}
    observations=OUT/'stability-observations-20260924.json';assert not observations.exists()
    observations.write_text(json.dumps(dict(closure=pin(BASE/'closed.json'),analysis=a,raw=raw),indent=2)+'\n')
    assert len(failed_controls)==10 and len(failed_cases)==22
    title='Control passes' if s['admitted'] else 'Control rejected'
    lines=['# Where repeatability with identical release binaries','',f'**{title}.** Both roles use selected Core `672e5f30` / Data `065b7a7f`; no optimization candidate is measured.', '',
           f"All correctness, identity and resource checks pass. {len(failed_controls)} of 225 repeatability controls fail; {len(failed_cases)} of 220 cases exceed the symmetric five-percent middle/outer-pair bound.",'',
           '| Sum of case means | Outer pair ms | Middle pair ms | Middle / outer | Pair bound passes |','|---|---:|---:|---:|---|']
    for name,row in s['scopes'].items():
        lines.append(f"| {name} | {row['outer']['value']*1000:.6f} | {row['middle']['value']*1000:.6f} | {row['ratio']['value']:.6f} | {row['passed']} |")
    lines += ['', 'The large-nonscalar-false case takes 22.230238, 14.102600, 14.389612 and 23.426960 microseconds across the four identical-binary processes. Its middle/outer ratio is 0.624046, a false 37.6% improvement. All three measured 60-sample blocks per process retain the same broad level; one isolated spike cannot explain that case. This does not identify compilation, allocation/layout or another runtime mechanism as the cause. The previous vector-last problem is absent at its historical scale in this distinct run (four-process max/min 1.0762), without overturning the earlier rejection.', '', 'All four fresh CPU2 processes use ordinary .NET 10.0.8 with the same consumer and product hashes. Each prepares all 220 valid cases, then completes 600 warmup samples for every case before starting 180 measured samples per case. Batches and the complete CPUExecutionProvider.Where boundary are fixed. The original 122 cases retain their order and partitions; 98 added cases cover the new numerical boundaries and broadcast runs.', '',
              'All inputs, exact result bits, OpResult metadata, full input stores, distinct outputs and held-output ownership pass. All clocks, including warmups and pauses, remain retained. Scores average every measured clock with equal process weights. The thirteen 60-sample blocks per case are descriptive only; none replaces the frozen score.', '',
              f"All seven jobs, {a['resources']:,} resource observations, {a['setup_count']:,} setups and {s['sample_clocks']:,} clocks pass. There are {s['public_calls']:,} complete public calls, including {s['measured_public_calls']:,} measured calls. Peak RSS is {a['peak_rss']:,} bytes.",'',
              f"Consumer SHA-256: `{a['consumer']['sha256']}`. Closure: `{pin(BASE/'closed.json')['sha256']}`. Tools were frozen at `2a0b9e7d`; owner 897724 / birth 1790217849.76 and all descendants are terminal.",'',
              '[Every case](stability-cases-20260924.csv), [all phase blocks](stability-phase-blocks-20260924.csv), and [exact scoring plus raw-file identities](stability-observations-20260924.json) are published. All 686,400 raw clocks remain in `artifacts/parakeet-dense-scalar-where-stability-amd-20260924/clocks.csv` and the verified collection archive. Keeping one retained clock export avoids another large tracked duplicate.','',
              ('This control permits a distinct candidate comparison with the exact same consumer, census, phase counts and batches. It does not establish a candidate gain or application improvement.' if s['admitted'] else 'This control does not admit the measurement protocol for candidate scoring. Preserve the failure and investigate it before making a speed claim; do not retry the unchanged control or use favorable blocks. The candidate remains numerically qualified and unmeasured.'),'',
              'Root product and BENCHMARK.md remain the fully qualified selected release.','']
    report=OUT/'stability-20260924.md';assert not report.exists();report.write_text('\n'.join(lines))
    print(json.dumps(dict(report=pin(report),closure=pin(BASE/'closed.json'),stability_admitted=s['admitted'],failed_controls=len(failed_controls),failed_cases=len(failed_cases))))


if __name__=='__main__':main()
