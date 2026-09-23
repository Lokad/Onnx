"""Publish every M39 clock and the complete rejected admission verdict."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-wide-per-call-screen-amd-20260923'


def read(p):return json.loads(p.read_text())
def pin(p):
    with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())


def main():
    assert pin(BASE/'closed.json')['sha256']=='66a3f377871e962d659d37f0b1f878107028ca31c23121d2466ffd0f0efecee7'
    closure=read(BASE/'closed.json');assert closure['passed'] and not closure['admitted']
    for name,wanted in closure['files'].items():assert pin(BASE/name)==wanted,name
    v=read(BASE/'analysis.json');state=read(BASE/'collected/identity.json')
    records=[];preparation=[];fixtures=read(BASE/'bundle/fixtures/result.json')['entries']
    for run in state['runs'][3:]:
        result=read(BASE/'collected'/run['name']/'result.json')
        for row in result['rows']:
            shared={key:row[key] for key in ['index','name','node','m','reduction','columns']}
            for clock in row['clocks']:
                records.append(dict(process=run['name'],role=result['role'],**shared,**clock,frequency=result['frequency'],seconds=clock['ticks']/result['frequency']))
            preparation.append(dict(process=run['name'],**shared,ticks=row['preparationTicks'],frequency=result['frequency'],seconds=row['preparationTicks']/result['frequency']))
    assert len(records)==5760 and len(preparation)==48
    for name,rows in [('screen-clocks-20260923.csv',records),('screen-preparation-20260923.csv',preparation)]:
        with (OUT/name).open('w',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (OUT/'screen-observations-20260923.json').write_text(json.dumps(dict(closure=pin(BASE/'closed.json'),**v),indent=2)+'\n')
    lines=['# Parakeet wide per-call MatMul: rejected','',
        '**The candidate is rejected.** Across the nine M>=64 cases, complete public',
        'MatMul calls are **5.14% slower**, missing the required 10% reduction. All',
        '28 repeatability controls pass, but all four performance gates fail. The',
        'selected product and BENCHMARK.md figures remain unchanged. No full-application',
        'candidate campaign or favorable rerun follows this result.','',
        '| Rows | Reduction | Columns | Current ms | Candidate ms | Candidate/current | <=5% regression |',
        '|---:|---:|---:|---:|---:|---:|:---:|']
    for row,fixture in zip(v['rows'],fixtures,strict=True):
        lines.append(f"| {fixture['m']} | {fixture['k']} | {fixture['n']} | {row['current']['value']*1000:.3f} | {row['candidate']['value']*1000:.3f} | {row['ratio']['value']:.4f} | {'Pass' if row['passed'] else 'Fail'} |")
    lines += ['', '| Aggregate (sum of case means) | Current ms | Candidate ms | Candidate/current |',
        '|---|---:|---:|---:|']
    for name,label in [('wide9','Nine M>=64 cases'),('all12','All twelve cases')]:
        row=v['scopes'][name];lines.append(f"| {label} | {row['current']['value']*1000:.3f} | {row['candidate']['value']*1000:.3f} | {row['ratio']['value']:.4f} |")
    lines += ['','All four fresh processes used CPU 2 and ordinary runtime defaults. Each had',
        '60 warmups and 60 measured calls per fixture. The clock includes validation,',
        'clearing, scratch rental/return, packing, consumption and tails. All 5,760',
        'clocks (2,880 measurements), 48 setup records, complete output checks and',
        '315 resource samples are retained. Peak owned RSS: 323,956,736 bytes.',
        'Owner 814618 (birth 1790156924.58) and all seven jobs terminated successfully.','',
        'Same-role aggregate spreads are at most 1.00353; the largest individual-case',
        'spread is 1.04991. The numerical qualification remains valid, but it provides',
        'no reason to promote this slower dispatch. The M51 route also moves despite',
        'unchanged eligibility, so no source-level claim of unchanged performance is',
        'made for that control. This screen does not identify the cause of regression.','',
        'A separate observation motivates inspecting the existing packing threshold:',
        'current M51 takes 18.873/18.582 ms for the two wide projections, whereas',
        'current M106 takes 11.777/12.111 ms despite more rows. The former does not',
        'pack per call; the latter does. This is a new hypothesis, not evidence that',
        'changing the threshold will improve complete transcription.','',
        'Evidence: `artifacts/parakeet-wide-per-call-screen-amd-20260923`.',
        'Closure: `66a3f377871e962d659d37f0b1f878107028ca31c23121d2466ffd0f0efecee7`.',
        '[Every clock](screen-clocks-20260923.csv), [setup](screen-preparation-20260923.csv),',
        '[exact gates and identities](screen-observations-20260923.json).','']
    (OUT/'screen-20260923.md').write_text('\n'.join(lines))
    print(json.dumps(dict(report=pin(OUT/'screen-20260923.md'),clocks=len(records),admitted=False)))


if __name__=='__main__':main()
