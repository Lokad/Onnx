"""Publish the closed first vector Sigmoid screen, including all failed gates."""
import csv
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-vector-sigmoid-screen-amd-20260925'


def main():
    loader=importlib.util.spec_from_file_location('publication_helpers',TOOLS.parent/'weight-ownership-probe/run.py')
    helper=importlib.util.module_from_spec(loader);loader.loader.exec_module(helper)
    pin,read=helper.pin,helper.read
    closure=read(BASE/'closed.json');analysis=read(BASE/'analysis.json')
    assert closure['passed'] and closure['analysis']==pin(BASE/'analysis.json')
    assert closure['admitted']==analysis['admitted'] and analysis['passed']
    for name,wanted in closure['files'].items():assert pin(BASE/name)==wanted,name
    assert analysis['diagnostic_only'] and not analysis['release_admitted'] and analysis['no_model_execution']
    assert len(analysis['rows'])==46 and len(analysis['controls'])==94
    paths=[TOOLS/('screen-20260925'+suffix) for suffix in ['.md','.json','.csv']]
    assert not any(p.exists() for p in paths)
    weighted=analysis['corpus_weighted'];current=weighted['current']['value'];candidate=weighted['candidate']['value']
    ratio=weighted['ratio']['value'];controls=sum(r['passed'] for r in analysis['controls']);cases=sum(r['passed'] for r in analysis['rows'])
    payload=dict(closure=pin(BASE/'closed.json'),**analysis)
    paths[1].write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n',encoding='utf8')
    with paths[2].open('x',newline='',encoding='utf8') as stream:
        writer=csv.writer(stream);writer.writerow(['case','current_seconds','candidate_seconds','candidate_over_current','within_5_percent'])
        for row in analysis['rows']:writer.writerow([row['name'],row['current']['value'],row['candidate']['value'],row['ratio']['value'],row['passed']])
    lines=['# Vector Sigmoid: complete public-call screen, 25 September 2026','',
        f"**{'Admitted for full model qualification' if analysis['admitted'] else 'Rejected by the prospective screen'}.** "
        f'The corpus-frequency-weighted sum is {current:.9f} s for isolated M78 and {candidate:.9f} s for the candidate '
        f'({ratio:.6f}×, {100*(1-ratio):.4f}% reduction). These are synthetic-value operator timings, not complete transcription latency.','',
        f'All {analysis["samples"]:,} samples are retained: {analysis["warmup_samples"]:,} warmups and '
        f'{analysis["measured_samples"]:,} measured batches across four fresh processes. '
        f'{controls}/94 repeatability controls and {cases}/46 per-case regression gates pass.','',
        '| Gate | Result |','| --- | --- |']
    lines += [f"| {g['name']} | {'pass' if g['passed'] else 'fail'} |" for g in analysis['gates']]
    lines += ['', 'The common consumer covers all 38 observed shapes, weighted by all 1,920 encoder Sigmoid calls '
        'per corpus, plus eight fallback/layout cases. The 600 rounds over all cases precede the 180 measured rounds. '
        'Every mean uses every measured batch with exact rational arithmetic; no trimming or timing correction is applied.', '',
        'Inputs use deterministic values `(i % 257 - 128) / 16`, not captured activations. The complete public call '
        'includes materialization, validation, allocation and arithmetic; fixture preparation and result checking are outside the clock. '
        'The unchanged fallback/double paths have their own gates. ORT contributes the previously observed shape census; '
        'there is no new ORT operator timing in this screen.', '',
        '| Process | Weighted seconds |','| --- | ---: |']
    lines += [f"| {name} | {value['value']:.9f} |" for name,value in analysis['processes'].items()]
    failed_controls=[r for r in analysis['controls'] if not r['passed']]
    failed_cases=[r for r in analysis['rows'] if not r['passed']]
    if failed_controls or failed_cases:
        lines += ['', '| Failed check | Ratio | Limit |','| --- | ---: | ---: |']
        lines += [f"| Repeatability: {r['role']} / {r['case']} | {r['ratio']['value']:.6f} | 1.10 |" for r in failed_controls]
        lines += [f"| Regression: {r['name']} | {r['ratio']['value']:.6f} | 1.05 |" for r in failed_cases]
    lines += ['', 'Numerical, input-ownership, held-output, exact-product, resource and process-accounting checks pass. '
        f"Peak process-tree RSS is {max(r['peak_rss'] for r in analysis['resources']):,} bytes over "
        f"{sum(r['samples'] for r in analysis['resources']):,} resource samples. "
        'CPU accounting retains its limitation for exited short-lived processes.', '',
        ('Next: complete native/model qualification in both execution modes, then the original full Parakeet application comparison. '
         'A passing operator screen does not establish the required 3% application gain.' if analysis['admitted'] else
         'Stop this candidate. Do not run full model/application qualification, retry this unchanged screen or substitute ISA, '
         'polynomial, compiler-flag or fusion variants. Retain this result and use the existing complete profile to narrow the next diagnosis.'), '',
        'The qualified root product and `BENCHMARK.md` stay unchanged. M78’s independent short-e5 admission failure remains open.', '',
        '[All case means and gates](screen-20260925.csv), [complete analysis](screen-20260925.json), '
        '[prospective protocol](../vector-sigmoid-screen/README.md), [numerical qualification](contracts-20260925.md), '
        '[exact M78/ORT diagnosis](../packed-final-row-profile-results/diagnosis-20260925.md). '
        'Raw clocks, resource samples, both runtime binaries and process states are retained under '
        '`artifacts/parakeet-vector-sigmoid-screen-amd-20260925`.', '',
        f"Closure SHA-256: `{pin(BASE/'closed.json')['sha256']}`."]
    paths[0].write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(published=[str(p.relative_to(ROOT)) for p in paths],admitted=analysis['admitted'],ratio=ratio)))


if __name__=='__main__':main()
