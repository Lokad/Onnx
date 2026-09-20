"""Render the full conditioned result without changing its declared decision."""
from pathlib import Path
from common import CORE,pin,read,write

def render(base,payload,audit):
    root=Path(__file__).resolve().parents[3];directory=root/'tests/e5/layernorm-conditioned'
    report=directory/'results-20260920.md';observations=directory/'observations-20260920.json'
    assert not report.exists() and not observations.exists()
    write(observations,{k:v for k,v in audit.items() if k!='births'})
    lines=['# Conditioned complete LayerNorm banks — September 20, 2026','',
        f"**{audit['verdict']}**. Every complete output, held result, input, parameter and resource check passes. This component result does not establish a complete-model or native ORT speedup.",'',
        'This distinct protocol follows the [inconclusive fixed-warmup result](../layernorm-bank/results-20260920.md). It preserves all nine banks, four fresh sequential workers and every original control/gain/regression threshold. Only prospective conditioning, batch size and measured cycle count change; no old result is relabeled.','',
        '| Bank | Product ms | Copy A ms | Copy B ms | Wide ms | Wide / Product | Wide / Copy A | Wide / Copy B | Controls | Gain / regression |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|']
    for c in audit['cases']:
        m=c['means_ms'];r=c['candidate_ratios'];lines.append(f"| {c['name']} | {m['Product']:.6f} | {m['CopyA']:.6f} | {m['CopyB']:.6f} | {m['Wide']:.6f} | {r['Product']:.6f} | {r['CopyA']:.6f} | {r['CopyB']:.6f} | {'pass' if c['controls_passed'] else 'fail'} | {'pass' if c['gain_passed'] and c['no_regression_passed'] else 'fail'} |")
    lines+=['','Each mean is milliseconds for all 25 complete normalizations in a bank, averaging four equal-sized visits. Statistics and output stores are timed; loading, hashing and validation are outside. Copy A and Copy B invoke the same unchanged source copy. Product invokes the actual archived kernel. [Complete observations](observations-20260920.json) retain every visit, distribution, allocation, GC and resource result.','',
        'Five real banks use the captured e5 inputs for 8/30/padded128/128/512. Four thirty-row diagnostic banks map column j to j modulo 384 at widths 383/385, with and without bias. Kernel source and all original inputs remain byte-identical to the closed AMD arithmetic/code proof.','',
        f"All {audit['measured_samples']:,} measured batches, {audit['warmup_samples']:,} conditioning batches and {audit['first_calls']} first calls are retained. Conditioning stops after the first complete four-role cycle whose summed kernel time reaches three seconds. Every cutoff is independently recomputed from raw integer ticks, including the preceding cycle. This fixed work-time budget is not convergence selection or proof of a particular JIT tier.",'',
        'Each bank then has 96 measured cycles with four rotating positions. Samples repeat 512/128/32/32/8 complete banks for the real cases and 128 for each diagnostic. There is no sample exclusion, forced GC, profiler or runtime override. First calls and conditioning remain separate from measured means.','',
        'Unchanged screens require duplicate means within 1% overall and 2% per worker; at least 2% candidate improvement against Product and both copies at 30/padded128/128; no regression above 1% overall or 2% per worker on any bank. These engineering screens are not confidence intervals. A pass permits subsequent product integration and full model qualification, without doing either automatically.','',
        'Every real output matches its original captured bits. All candidate/copy diagnostic outputs match actual product bits; sixteen saved diagnostic arrays pass independent centered-double scalar normalization at 1e-5. Actual first outputs stay held through every later call. Complete input and parameter hashes remain unchanged.','',
        'Linux .NET 10.0.8, AMD EPYC 9V74, logical CPU 2 inherited before CLR startup; supervisor CPU 0. Eight-float ordinary vectors and hardware AVX-512 are verified. Native ORT is not loaded. Core SHA256 `'+CORE+'`.','',
        '| Visit | Seconds | Peak sampled RSS bytes | Minimum available bytes | Foreign CPU fraction | Steal fraction |',
        '|---|---:|---:|---:|---:|---:|']
    for r in audit['resources']:lines.append(f"| {r['visit']} | {r['seconds']:.3f} | {r['peak_rss']:,} | {r['minimum_available']:,} | {r['foreign']['foreign_cpu_fraction']:.6g} | {r['steal_fraction']:.6g} |")
    lines+=['','All original PID/creation-time identities are terminal. Fixed 600-second, 3-GiB sampled group-RSS, 2-GiB available-memory, 2% observed foreign-CPU and 0.5% guest-steal limits pass. Snapshots miss some short-lived work and cannot control hypervisor neighbors.','',
        f"Frozen generator source `{read(payload/'bundle.json')['source']}`; bundle SHA256 `{pin(payload/'bundle.json')['sha256']}`; audit SHA256 `{pin(base/'audit.json')['sha256']}`.",
        'Complete generated-source provenance, raw observations, diagnostic arrays and receipts are retained under `artifacts/e5-layernorm-conditioned-20260920`. The preceding failed controls remain unchanged.','']
    with report.open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines))
    return [report,observations]
