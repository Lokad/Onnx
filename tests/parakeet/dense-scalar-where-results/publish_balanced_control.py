"""Publish the fixed balanced-control verdict and every case/block without raw duplication."""
import csv
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/parakeet-dense-scalar-where-balanced-control-amd-20260924'
HERE=Path(__file__).resolve().parent


def pin(path):
    with path.open('rb') as stream:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def read(path):return json.loads(path.read_text())


def write_csv(path,rows):
    with path.open('x',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    names=['balanced-control-20260924.md','balanced-control-cases-20260924.csv',
        'balanced-control-blocks-20260924.csv','balanced-control-observations-20260924.json']
    assert all(not (HERE/n).exists() for n in names)
    proof=read(BASE/'closed.json');assert proof['passed']
    for name,wanted in proof['files'].items():assert pin(BASE/name)==wanted,name
    analysis=read(BASE/'analysis.json');result=analysis['comparison']
    assert proof['stability_admitted']==analysis['stability_admitted']==result['admitted']
    assert analysis['consumer']['sha256']=='bfd850bcde3d38a0dd29448462780f74312c642218bb90ee632a8538270e0c51'
    assert analysis['identical_binary_control'] and not analysis['candidate_measured']
    assert analysis['sample_clocks']==686400 and analysis['public_calls']==232929840
    cases=[]
    for row in result['rows']:
        cases.append(dict(index=row['index'],name=row['name'],partition=row['partition'],
            outer_seconds=row['outer']['value'],middle_seconds=row['middle']['value'],
            middle_over_outer=row['ratio']['value'],pair_passed=row['passed']))
    write_csv(HERE/names[1],cases)
    blocks=[];raw={};census=read(BASE/'bundle/cases.json')
    for name in result['processes']:
        path=BASE/'collected'/name/'result.json';value=read(path)
        raw[name]={item:dict(path=(BASE/'collected'/name/item).relative_to(ROOT).as_posix(),identity=pin(BASE/'collected'/name/item))
                   for item in ['result.json','clocks.jsonl','setups.jsonl']}
        for row,case in zip(value['rows'],census,strict=True):
            assert row['name']==case['name'] and len(row['clocks'])==780
            for block in range(13):
                part=row['clocks'][block*60:(block+1)*60]
                ticks=sum(c['ticks'] for c in part)
                mean=F(ticks,60*value['frequency']*row['batch'])
                blocks.append(dict(process=name,index=row['index'],name=row['name'],partition=case['partition'],
                    block=block,warmup=block<10,first_iteration=block*60,last_iteration=(block+1)*60-1,
                    samples=60,batch=row['batch'],ticks=ticks,frequency=value['frequency'],mean_seconds=float(mean)))
    assert len(cases)==220 and len(blocks)==11440
    write_csv(HERE/names[2],blocks)
    observations=dict(closed=pin(BASE/'closed.json'),analysis=pin(BASE/'analysis.json'),raw=raw,
        comparison=result,summary_files={n:pin(HERE/n) for n in names[1:3]},
        note='Every case and all thirteen blocks per process are retained. Blocks are descriptive; the fixed score uses every measured clock with equal process weights.')
    with (HERE/names[3]).open('x',encoding='utf8') as stream:json.dump(observations,stream,indent=2);stream.write('\n')
    failed_controls=[r for r in result['controls'] if not r['passed']]
    failed_pairs=[r for r in result['rows'] if not r['passed']]
    admitted=result['admitted']
    lines=['# Balanced-warmup identical-binary Where control','',
        '**Control '+('admitted' if admitted else 'rejected')+'.** Both roles use the same selected products and consumer; no optimization candidate is measured.','',
        f'{len(failed_controls)} of 225 repeatability checks fail; {len(failed_pairs)} of 220 case-pair checks fail. The five aggregate pair checks are included in the admission decision.','',
        '| Sum of case means | Outer pair ms | Middle pair ms | Middle / outer | Pair bound passes |',
        '|---|---:|---:|---:|---|']
    for scope,row in result['scopes'].items():
        lines.append(f'| {scope} | {1000*row["outer"]["value"]:.6f} | {1000*row["middle"]["value"]:.6f} | {row["ratio"]["value"]:.6f} | {row["passed"]} |')
    lines+=['',
        'The prospective limits are unchanged: four-process aggregate max/min <= 1.10, each case <= 1.20, and every middle/outer pair ratio within [20/21, 21/20]. All 220 cases and all five aggregates are required. Scores retain all 180 measured clocks per case and equal process weights.','',
        'Each process prepares the same inputs once, performs 600 round-robin warmup rounds over all 220 cases, then completes 180 measured samples per case in original case order. The exact qualified shared timing method encloses the complete public Where calls and result writes. Managed flags are empty. No adaptive warmup, clock trimming or sample replacement occurred.','',
        f'All correctness, identity and resource checks pass: {analysis["resources"]:,} resource observations, 880 setups, 686,400 clocks and 232,929,840 complete calls, including 53,753,040 within measured intervals. Peak process-tree RSS is {analysis["peak_rss"]:,} bytes. Exact output bits, metadata, input stores and held/output ownership match the qualified cases.','',
        '[Every case](balanced-control-cases-20260924.csv), [all phase blocks](balanced-control-blocks-20260924.csv), and [exact scores with raw-file identities](balanced-control-observations-20260924.json) are retained. All raw clocks remain in the verified local collection and archive. No additional tracked raw-clock copy is created.','',
        f'Closure: `{pin(BASE/"closed.json")["sha256"]}`. Tools frozen at `bd02a673`. Consumer `bfd850bc` (59,392 bytes), Core `672e5f30`, Data `065b7a7f`. Collection: `artifacts/parakeet-dense-scalar-where-balanced-control-amd-20260924`.','',
        ('This control permits preparation of a separately frozen candidate comparison. It establishes no candidate speed gain.' if admitted else
         'Park M59 after this fixed rejection and return to complete Parakeet profiling and the separately reviewed bounded packing hypothesis. Preserve every failed check; do not repeat this unchanged design, drop cases or relax thresholds.'),'',
        'Root product and BENCHMARK.md remain the qualified selected release.','']
    with (HERE/names[0]).open('x',encoding='utf8') as stream:stream.write('\n'.join(lines))
    print(json.dumps(dict(stability_admitted=admitted,reports={n:pin(HERE/n) for n in names})))


if __name__=='__main__':main()
