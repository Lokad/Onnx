"""Summarize the independently closed AMD capture without changing timing tables."""
from common import ROOT, pin, read, save, verify, rel
from pathlib import Path

BASE = ROOT / 'artifacts/pyannote-amd-profile-v3-20260922'
OUTPUT = Path(__file__).resolve().parent


def main():
    closed = read(BASE / 'closed.json'); assert closed['passed']; verify(closed['files'])
    assert pin(BASE / 'closed.json')['sha256'] == 'cc9b31a0cb9abe795c24789731ff462515b394133058f4ea6a62b30fd3f3b2cc'
    analysis = read(BASE / 'analysis.json'); assert analysis['passed'] and analysis['calls'] == 48
    state = read(BASE / 'collected/identity.json')
    assert all(r['preflight']['disk'] >= 3*1024**3 for r in state['runs'])
    # Every ready birth is a separate metadata cross-check; ownership keeps exact births.
    birth_deltas = {r['name']: abs(r['ready']['birth_milliseconds']/1000-r['processes']['target']['birth']) for r in state['runs']}
    assert max(birth_deltas.values()) < 1.1
    names = [
        ('Packed three-row matrix kernel', 'MathOps.mm_unsafe_vectorized_intrinsics_3x4packed('),
        ('Tiled convolution caller', '.RunTiledBatchFloat('),
        ('LSTM provider', 'CPUExecutionProvider.Lstm('),
        ('Ordered LSTM projection', 'CPUExecutionProvider.LstmProjectOrdered('),
        ('Memory copy', 'SpanHelpers.Memmove('),
        ('Packed two-row remainder kernel', 'MathOps.mm_unsafe_vectorized_intrinsics_2x4packed_bump('),
        ('Transpose', '.TransposeInto('),
    ]
    rows = []
    for label, fragment in names:
        values = []
        for capture in analysis['diagnostics']:
            matching = [r for r in capture['exclusive'] if r['marker']=='dialogue-30s' and fragment in r['method']]
            assert matching
            assert len({r['method'] for r in matching}) == 1
            seconds = sum(r['seconds'] for r in matching)
            values.append(dict(seconds=seconds, share=100*seconds/capture['selected_seconds']['dialogue-30s'], buckets=matching))
        rows.append(dict(label=label, captures=values))
    table = ['| Exclusive leaf | Capture A share | Capture B share |', '|---|---:|---:|']
    table += [f"| {r['label']} | {r['captures'][0]['share']:.2f}% | {r['captures'][1]['share']:.2f}% |" for r in rows]
    timing = ['| Fixture | Control mean seconds | Capture A mean | Capture B mean |', '|---|---:|---:|---:|']
    for row in analysis['observations']:
        timing.append('| '+row['name']+' | '+' | '.join(f"{row['roles'][role]['wall_mean']:.6f}" for role in ['control','sampled-a','sampled-b'])+' |')
    observations = dict(passed=True, closure=pin(BASE / 'closed.json'), analysis=pin(BASE / 'analysis.json'),
        leaves=rows, observations=analysis['observations'], coverage=[dict(name=d['name'], coverage=d['coverage'], exports=d['exports']) for d in analysis['diagnostics']],
        resources=analysis['resources'], remote_identities=analysis['remote_identities'], local_identities=analysis['local_identities'],
        birth_metadata_deltas_seconds=birth_deltas, preflight_tmpfs_at_least_3gib=True)
    assert not (OUTPUT / 'observations-20260922.json').exists() and not (OUTPUT / 'results-20260922.md').exists()
    save(OUTPUT / 'observations-20260922.json', observations)
    samples = sum(r['samples'] for r in analysis['resources']); peak = max(r['peak_rss'] for r in analysis['resources'])
    report = f'''# Selected Pyannote complete-request attribution on AMD

The packed three-row matrix kernel accounts for **53.33–54.17%** of sampled
full-request thread time. The tiled convolution caller contributes another
**15.70–15.96%**. These results confirm convolution as the first remaining
optimization target on the actual AMD EPYC 9V74 VM. LSTM execution and ordered
projection together account for 13.54–14.50%.

{chr(10).join(table)}

These are exclusive sampled managed-thread weights under complete public
request markers. Inlining can charge helper work to the caller; the tiled
caller share does not isolate bias, copying or tensor construction. Exporter
CPU_TIME labels remain labels. Process CPU is measured and reported separately.

All **48 public requests** pass native, input, repeat and held-output checks
and preserve the earlier selected AMD results exactly. Each capture contains
three full requests and three calls to each crop, with three marker intervals
per fixture on the correct target thread. Both exports agree on every event.
Full-request sampled/wall totals are 51.435384/51.430557 seconds and
51.726950/51.722192 seconds; process CPU totals are 49.040720 and 49.326450
seconds. Every original fixture coverage bound passes.

{chr(10).join(timing)}

Captured full-request means are 10.24% and 10.87% above the diagnostic control.
These differences combine instrumentation overhead and process variation;
they are not used to correct samples or estimate unprofiled component costs.
This is not a fresh ORT comparison. The accepted matched application result
remains 15.466 seconds versus Microsoft ORT 8.952 seconds (1.728 ratio).

## Fixed runtime and execution

Core e9c87932 / Data 85d166b5 are the exact selected binaries, whose compiled
methods match integrated root production. Product bytes are unchanged. The
consumer changes only thread-ID lookup for Linux: 157 of 158 existing methods
and all public declarations match, with two platform imports added. Main
request code, barriers, markers and numerical assertions remain identical.
The [protocol](README.md) preserves preparation failures and their corrections.
The original report generator also failed on a zero-weight second bucket for
TransposeInto; report_v2.py sums and retains every matching bucket. Capture
and audit files remain unchanged.

One control and two sampled processes run sequentially on .NET 10.0.8. Target
CPU 2 affinity is inherited before CLR; collector and monitor use CPU 0. Every
observed native thread has its role's affinity. The pinned dotnet-trace
10.0.745401 collector uses the original providers and post-warmup barrier.
Local preparation session 87464 and trace-export/audit commands exit zero.
Remote supervisor 665095 / birth 1790046518.91 and all five target/collector
processes are terminal with successful recorded exits before collection.

All **{samples} resource samples** pass, with peak aggregate RSS
**{peak:,} bytes**. Each pair actually starts with more than 3 GiB tmpfs free.
The resource bounds, 1.1-second Linux birth-metadata cross-check and exact
PID/start-time ownership are retained in the complete observations. Evidence
is streamed to Windows after termination; the VM root disk receives no payload
or result archive. Collection verifies 78 files and unchanged frozen inputs.

Closure: cc9b31a0cb9abe795c24789731ff462515b394133058f4ea6a62b30fd3f3b2cc.
Collected archive: b4eac3bf497d612ab184504908b3fad08d641f7362871b8e6eb2a9a84ef7c90d.
Artifact: artifacts/pyannote-amd-profile-v3-20260922.

## Next experiment

Review a reduction-blocked consumer specifically for the current three-row
convolution kernel. Preserve each output's ascending FMA recurrence, existing
packing, tails and scalar/experimental fallbacks. Qualify all 22 actual tile
geometries, including the six strided 1x1 geometries omitted by the original
standalone row-group probe. Any timing must include clearing, packing and
multiplication before progressing to complete application comparisons.

The [earlier e5 blocking trial](../../e5/projection-reduction-timing/results-20260920.md)
regressed all five banks with stable controls. It used twelve/eight-row AVX-512
prepared projections. That failure remains; it supplies no evidence of a gain
for this different three-row convolution workload. The prospective convolution
experiment must stand on its own correctness and measured results. Neither
source structure nor this profile proves a cache bottleneck or a speedup.
'''
    (OUTPUT / 'results-20260922.md').write_text(report, encoding='utf8')
    print(dict(report=pin(OUTPUT / 'results-20260922.md'), observations=pin(OUTPUT / 'observations-20260922.json')))


if __name__ == '__main__': main()
