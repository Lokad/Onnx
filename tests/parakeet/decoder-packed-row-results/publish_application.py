"""Publish the single complete transcription experiment, preserving its verdict."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-decoder-packed-row-app-amd-20260927'


def read(p):return json.loads(p.read_text(encoding='utf8'))
def pin(p):
    with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def csvfile(path,rows):
    with path.open('x',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    proof=read(BASE/'closed.json');value=read(BASE/'analysis.json');payload=read(BASE/'payload.json')
    assert proof['passed'] and value['passed'] and proof['analysis']==pin(BASE/'analysis.json')
    for name,wanted in proof['files'].items():assert pin(BASE/name)==wanted,name
    assert value['identities']==payload['identities']
    assert payload['identities']['current']['Lokad.Onnx.dll']['sha256']=='65f15a41764660af6166c9a965943b6f28cac0c9505117d0fd4beaa2687f9d03'
    assert payload['identities']['candidate']['Lokad.Onnx.dll']['sha256']=='af19b3b4429a07f7966b5e35ee04e8a31316f45991c300f3683a591caf5e9374'
    assert payload['identities']['current']['Lokad.Onnx.Data.dll']['sha256']=='da72ca547191de12a52f09bceeed25619a92a14b1bbff271a9e7d10515f40312'
    assert payload['identities']['candidate']['Lokad.Onnx.Data.dll']['sha256']=='da72ca547191de12a52f09bceeed25619a92a14b1bbff271a9e7d10515f40312'
    assert not payload['release_admitted'] and len(payload['failed_component_controls'])==2
    assert all(not r['passed'] for r in payload['failed_component_controls'])
    assert not value['root_product_changed']
    assert (value['timing_requests'],value['warmup'],value['measured'])==(480,120,360)
    perf=value['performance'];assert proof['admitted']==perf['admitted']
    assert len(perf['controls'])==63 and len(perf['gates'])==21
    corpus,=[r for r in value['table'] if r['is_corpus']]
    assert corpus['audio_seconds']==213.265 and len(value['table'])==21
    gain=1-corpus['candidate']['seconds']/corpus['current']['seconds']
    controls=sum(r['passed'] for r in perf['controls']);gates=sum(r['passed'] for r in perf['gates'])
    rows=[dict(name=r['name'],audio_seconds=r['audio_seconds'],current_seconds=r['current']['seconds'],
               candidate_seconds=r['candidate']['seconds'],ort_seconds=r['ort']['seconds'],
               candidate_over_ort=r['ratios_to_ort']['candidate'],
               candidate_over_current=r['candidate']['seconds']/r['current']['seconds']) for r in value['table']]
    clocks=[]
    for i,role in enumerate(['current','candidate','ort','ort','candidate','current']):
        name=f'timing-{i:02}-{role}'
        result=read(BASE/'collected'/name/'output/result.json')
        for index,r in enumerate(result['records']):
            clocks.append(dict(process=name,role=role,index=index,**{k:r[k] for k in
                ['name','phase','start_ticks','end_ticks','frequency','seconds']}))
    assert len(clocks)==480 and sum(r['phase']=='measured' for r in clocks)==360
    paths={suffix:OUT/('application-20260927'+suffix) for suffix in ['.md','.json','.csv','-clocks.csv']}
    assert not any(p.exists() for p in paths.values()),'Preserve existing publication'
    failed=[f"- Repeatability: {r['name']} / {r['role']}, {r['process_ratio']:.6f} > {r['limit']:.2f}."
            for r in perf['controls'] if not r['passed']]
    failed += [f"- Performance: {r['name']}, candidate/current root {r['candidate_over_current']:.6f} > {r['limit']:.2f}."
               for r in perf['gates'] if not r['passed']]
    failure_text=('\n\nFailed checks:\n\n'+'\n'.join(failed)) if failed else ''
    prose=f'''# Prepared single-row weights: complete Parakeet comparison

**The candidate {'passes' if perf['admitted'] else 'fails'} the prospective application admission.**
Complete transcription latency is {100*abs(gain):.3f}% {'lower' if gain>=0 else 'higher'} than current root.
Repeatability: {controls}/63 checks. Performance: {gates}/21 gates.
Candidate/ORT is **{corpus['ratios_to_ort']['candidate']:.6f}**; the independent
1.05 parity target is {'met' if perf['parity_target_met'] else 'not met'}.{failure_text}

| Twenty-clip corpus (213.265 s audio) | Seconds | Relative to ORT |
| --- | ---: | ---: |
| current root baseline | {corpus['current']['seconds']:.6f} | {corpus['ratios_to_ort']['current']:.6f} |
| Prepared-row candidate | {corpus['candidate']['seconds']:.6f} | {corpus['ratios_to_ort']['candidate']:.6f} |
| Microsoft ORT 1.29.0 | {corpus['ort']['seconds']:.6f} | 1.000000 |

Six fresh processes run current root, candidate, ORT, ORT, candidate, current root on AMD EPYC
9V74 CPU 2. Each uses one warmup and three measured passes over every clip:
480 requests, 120 warmups and 360 measurements. The common consumers,
per-request validation and exact-clock aggregation are unchanged. There is no
profiling, compilation or runtime override in this comparison. No clock is trimmed.

The prospective limits require at least 1% corpus gain, no clip more than 5%
slower, corpus process max/min <= 1.10 and per-clip max/min <= 1.20 for all engines.
All workers are terminal/code0; all {sum(r['samples'] for r in value['resources']):,}
resource samples pass. Peak owned RSS is {max(r['peak_rss'] for r in value['resources']):,} bytes.

The single product change reads the existing packed final-projection weights while
preserving arithmetic, reduction order, vector width and independently owned
outputs. [Complete model correctness](models-20260927.md) passes the original ORT
bound, byte-exact current/candidate tensors, decoder decisions/transcripts and
ownership checks in normal and AVX512-disabled execution. The
[original decoder observation](../decoder-projection-observation-results/README.md)
identified an available packed constant with an executed row-major kernel.

The [operator screen](screen-20260927.md) remains rejected: two repeatability
controls and two fallback regressions failed. The [per-call diagnosis](unmapped-calls-20260927.md)
rejects the first-call-only explanation and shows recovery over several calls.
Cache competition is an inference, not a measured hardware cause. This independent
application result does not change either verdict or establish universal fallback
performance equivalence.

{'Next: resolve or explicitly justify the retained fallback limitation, then require shared/e5, Pyannote, graph and actual-root/package qualification before any source or BENCHMARK.md promotion.' if perf['admitted'] else 'This candidate fails application admission and is not promoted. Do not rerun this unchanged comparison, trim clocks or revise its gates. Use the measured result to choose the next causal investigation.'}
The current release and BENCHMARK.md stay unchanged. Parity remains a separate goal.

[Every case](application-20260927.csv), [all 480 clocks](application-20260927-clocks.csv),
and [all controls, identities and resources](application-20260927.json).
Closure: `{pin(BASE/'closed.json')['sha256']}`.
Raw evidence: `{BASE.relative_to(ROOT).as_posix()}`.
'''
    with paths['.json'].open('x',encoding='utf8') as f:
        json.dump(dict(closure=pin(BASE/'closed.json'),**value,release_admitted=False,
                       retained_failed_component_controls=payload['failed_component_controls']),f,indent=2,allow_nan=False);f.write('\n')
    csvfile(paths['.csv'],rows);csvfile(paths['-clocks.csv'],clocks)
    with paths['.md'].open('x',encoding='utf8') as f:f.write(prose)
    print(json.dumps(dict(admitted=perf['admitted'],gain=gain,candidate_over_ort=corpus['ratios_to_ort']['candidate'],
        controls=controls,gates=gates,release_admitted=False,report=pin(paths['.md']))))


if __name__=='__main__':main()
