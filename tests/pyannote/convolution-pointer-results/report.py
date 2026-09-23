"""Publish the closed comparison and every original integer clock."""
import csv
from fractions import Fraction
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'tests/pyannote/convolution-pointer-screen-amd'))
from protocol import pin,read,save
from score import ORDER,validate_and_score

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/pyannote-convolution-pointer-screen-amd-20260923'
OUT=Path(__file__).resolve().parent


def f(value):return Fraction(value['numerator'],value['denominator'])


def main():
    assert not (OUT/'results-20260923.md').exists()
    closure=read(BASE/'closed.json');assert closure['passed']
    for name,wanted in closure['files'].items():assert pin(BASE/name)==wanted,name
    analysis=read(BASE/'analysis.json');payload=read(BASE/'payload.json')
    reports={name:read(BASE/'collected'/name/'result.json') for name in ORDER}
    fixtures=read(BASE/'bundle/fixtures/result.json');reference=read(BASE/'bundle/reference.json')
    score=validate_and_score(reports,fixtures,reference)
    for key,value in score.items():assert analysis[key]==value,key
    assert closure['admitted']==score['admitted']
    for kind,filename in [('observations','clocks-20260923.csv'),('preparation','preparation-20260923.csv')]:
        fields=['process',*reports[ORDER[0]][kind][0].keys()]
        with (OUT/filename).open('x',encoding='utf8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for name in ORDER:
                for row in reports[name][kind]:writer.writerow(dict(process=name,**row))
    save(OUT/'observations-20260923.json',dict(closure=pin(BASE/'closed.json'),identities=payload['job_details'],
        driver=payload['driver'],analysis=analysis,clocks=pin(OUT/'clocks-20260923.csv'),preparation=pin(OUT/'preparation-20260923.csv')))
    aggregate=analysis['rows'][0];p,c=f(aggregate['production']),f(aggregate['candidate'])
    controls=[r for r in analysis['controls'] if not r['passed']];gates=[r for r in analysis['gates'] if not r['passed']]
    lines=['# Row pointers with fixed spatial steps: complete-call screen','',
        '**Admitted for full product/application qualification.**' if analysis['admitted'] else '**Not admitted.**',
        f'The sum of all 108 prepared graph call means is {float(p):.9f} s for the',
        f'current product and {float(c):.9f} s for the candidate: ratio {float(c/p):.9f}.',
        f'This is {abs(1-float(c/p))*100:.4f}% {"lower" if c<=p else "higher"} component latency.',
        f'{len(controls)} of 32 repeatability controls and {len(gates)} of 12 speed gates fail.',
        'All numerical, ownership and resource checks pass. The root product is unchanged.','',
        '| Form | Eligible | Current ms | Candidate ms | Candidate / current |',
        '|---:|---|---:|---:|---:|']
    for row in analysis['rows'][1:]:
        lines.append(f'| {row["form"]} | {row["eligible"]} | {float(f(row["production"]))*1000:.6f} | {float(f(row["candidate"]))*1000:.6f} | {float(f(row["ratio"])):.6f} |')
    lines+=['','Each form keeps its original multiplicity over all three captured crops.',
        'Four fresh processes run current, candidate, candidate, current on AMD',
        'EPYC 9V74 CPU2 with .NET 10.0.8. Geometry fixes 1,074 iterations per pass;',
        'one warmup and three measured passes retain 17,184 clocks: 4,296 warmup',
        'and 12,888 measured. All 512 separate preparation clocks are retained.',
        'No calibration, exclusions, sample trimming, profiling or runtime overrides.','',
        'The unchanged caller includes assertions/dispatch recording, finite scans,',
        'scratch, conversions, kernel and graph epilogues. Hashing and journal IO',
        'are outside the timer. Separate preparation clocks cover graph creation',
        'and preparation. This is not an embedding/diarization application timer',
        'and supplies no new Microsoft ORT speed result.','',
        '| Process | Sum of call means s | Preparation of 32 nodes ms |',
        '|---|---:|---:|']
    for name in ORDER:lines.append(f'| {name} | {float(f(analysis["process_totals"][name]["total"])):.9f} | {float(f(analysis["preparation"][name]))*1000:.6f} |')
    lines+=['','Exact clock fractions enforce the original limits: process max/min ≤1.10',
        'aggregate and ≤1.20 per form; candidate/current ≤0.98 aggregate and ≤1.05',
        'for every eligible form. All 32 controls and 12 speed gates are mandatory.',
        'Fifteen prospective scorer tests pass. Strict process separation is also mandatory.']
    separation=analysis['process_separation']
    lines+=['',f'Strict process separation: {"PASS" if separation["passed"] else "FAIL"}.',
        f'Largest candidate process aggregate {float(f(separation["candidate_max"])):.9f} s must be strictly below smallest current aggregate {float(f(separation["current_min"])):.9f} s.',
        'These incremental criteria were frozen before any M28 timing; historical verdicts retain their original rules.']
    if controls or gates:
        lines+=['','Failed criteria:','']
        for label,rows in [('Repeatability',controls),('Speed',gates)]:
            for row in rows:lines.append(f'- {label}, {row.get("role","candidate")} form {row["form"] if row["form"] is not None else "aggregate"}: {float(f(row["ratio"])):.9f}, limit {float(f(row["limit"])):.2f}.')
    lines+=['',f'All {analysis["samples"]} resource observations pass; peak owned RSS is {analysis["peak_rss"]:,} bytes.',
        'All recorded process owners are terminal, and every collected file is verified.',
        f'Current Core: `{payload["job_details"]["production"]["core"]["sha256"]}`.',
        f'Candidate Core: `{payload["job_details"]["candidate"]["core"]["sha256"]}`.',
        f'Independent closure: `{pin(BASE/"closed.json")["sha256"]}`.','',
        '[All call clocks](clocks-20260923.csv), [preparation clocks](preparation-20260923.csv),',
        '[all gates and resource summaries](observations-20260923.json),',
        '[AMD numerical qualification](numerics-20260923.md),',
        '[generated-code inspection](codegen-20260923.md).','',
        'Full product suites/package, Pyannote/Parakeet/shared/e5 regressions, both long meetings and a fresh matched application/ORT campaign remain mandatory before integration.' if analysis['admitted'] else 'No root integration or unchanged timing retry follows this rejected screen.',
        'Pyannote remains first, Parakeet second; Whisper is deferred.']
    (OUT/'results-20260923.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(admitted=analysis['admitted'],report=pin(OUT/'results-20260923.md'),ratio=float(c/p))))


if __name__=='__main__':main()
