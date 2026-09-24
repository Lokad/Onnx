"""Publish the single fixed M56 comparison, preserving its rejection and every clock."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'artifacts/parakeet-scalar-where-screen-amd-20260924'
OUT = Path(__file__).resolve().parent


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def main():
    assert pin(BASE/'closed.json')['sha256']=='fd8477677a34b9064fa5294d1834b91a59a2284b6bbfbb0ff470c612a2c4949b'
    proof=json.loads((BASE/'closed.json').read_text());assert proof['passed'] and not proof['performance_admitted']
    for name,wanted in proof['files'].items():assert pin(BASE/name)==wanted,name
    analysis=json.loads((BASE/'analysis.json').read_text());score=analysis['comparison']
    assert not score['admitted'] and len(score['controls'])==252 and len(score['rows'])==122
    failed_controls=[c for c in score['controls'] if not c['passed']]
    failed_rows=[r for r in score['rows'] if not r['passed']]
    assert len(failed_controls)==52 and len(failed_rows)==40
    for source,target in [('clocks.csv','screen-clocks-20260924.csv'),('setups.json','screen-setups-20260924.json')]:
        assert not (OUT/target).exists();shutil.copy2(BASE/source,OUT/target)
    report=OUT/'screen-observations-20260924.json';assert not report.exists()
    report.write_text(json.dumps(dict(closure=pin(BASE/'closed.json'),analysis=analysis),indent=2)+'\n',encoding='utf8')
    cases=OUT/'screen-cases-20260924.csv';assert not cases.exists()
    with cases.open('w',newline='',encoding='utf8') as stream:
        writer=csv.writer(stream);writer.writerow(['index','name','partition','current_seconds','candidate_seconds','candidate_current','case_gate'])
        for row in score['rows']:
            writer.writerow([row['index'],row['name'],row['partition'],row['current']['value'],row['candidate']['value'],row['ratio']['value'],row['passed']])
    lines=['# Uniform-mask Where: fixed complete-call comparison','',
        '**Candidate rejected.** The required stability and fallback-regression gates fail. No application run or integration follows this result.','',
        '| Sum of per-case means | Current ms | Candidate ms | Candidate/current |',
        '|---|---:|---:|---:|']
    labels={'target6':'Six actual Parakeet float cases','other_uniform42':'42 other uniform cases','fallback74':'74 fallback cases','all122':'All 122 cases'}
    for key in labels:
        v=score['scopes'][key];lines.append(f"| {labels[key]} | {v['current']['value']*1000:.6f} | {v['candidate']['value']*1000:.6f} | {v['ratio']['value']:.6f} |")
    lines += ['',
        'The target sum and its strict process separation pass. These descriptive averages do not qualify a speedup: only **200/252 stability controls** pass, and **40/122 cases** exceed the 5% regression bound. Candidate target sums differ by 1.6893x between fresh processes. The two current-release processes also fail 32 case-repeatability checks. No clock is discarded and no unchanged screen is repeated.','',
        'The candidate fallback sum is close to the selected sum, but that aggregate hides substantial individual regressions. For example, scalar-result-false is 0.111724 us versus 0.494072 us (4.4222x), x-reverse-false is 0.330318 us versus 2.617652 us (7.9246x), and raw-bool-0 is 0.199955 us versus 0.225783 us (1.1292x). The actual large capture-7-first mixed mask also exceeds the case bound at 1.0826x. Neither the large uniform gains nor the aggregate fallback result override these failures.','',
        'This screen does not capture generated code. Prior numerical code-generation workers emitted different tier inventories, so a JIT-tier explanation for these regressions remains a hypothesis. Exact preservation of the original fallback IL does not establish identical emitted code, promotion timing or first-call behavior. Inspect those mechanisms before another source change; do not compensate by changing thresholds after seeing clocks.','',
        'Four fresh CPU2 processes ran current/candidate/candidate/current with ordinary runtime settings and no profiler. All 122 valid cases from the qualified numerical census were included; only nine intentional exception cases were excluded. Each case used 60 warmup and 60 measured samples with the predeclared deterministic batch formula. Every sample includes complete public Where calls, allocation and owned output. Setup and correctness checks are outside timing. Equal process weighting and exact rational arithmetic retain every measured tick.','',
        f"All seven jobs, 385 resource observations, 488 setups and {score['sample_clocks']:,} sample clocks close successfully. There are {score['measured_clocks']:,} measured sample clocks, {score['public_calls']:,} public calls, and {score['measured_public_calls']:,} measured public calls. Peak RSS is 332,197,888 bytes. Output bits, shapes, complete input stores/guards, independent ownership and held outputs all pass. Owner888743/birth1790210791.22 and every descendant are terminal, code0.",'',
        'Selected Core672e5f30/Data065b7a7f; candidate Core70275a50/Data7d26bf8c. Consumer SHA-256: d710d3f7acebc7badaaa7130578f335477383ba7a68230cf369a0a5a952ab9a1. Frozen tools:a71490ac. Archiveb97c017a,stagea8bf0a61,payloadf976723b.','',
        'Closure:fd8477677a34b9064fa5294d1834b91a59a2284b6bbfbb0ff470c612a2c4949b. Complete local evidence:artifacts/parakeet-scalar-where-screen-amd-20260924.','',
        '[All 122 cases](screen-cases-20260924.csv), [every clock](screen-clocks-20260924.csv), [all setups](screen-setups-20260924.json), and [exact controls and identities](screen-observations-20260924.json) are retained. Root source81f75c38 and BENCHMARK.md remain the qualified release. These component results establish no new complete-transcription time or ORT ratio.','']
    target=OUT/'screen-20260924.md';assert not target.exists();target.write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps(dict(closure=pin(BASE/'closed.json'),failed_controls=len(failed_controls),failed_cases=len(failed_rows),report=pin(target))))


if __name__=='__main__':main()
