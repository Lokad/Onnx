"""Report the current complete-request profile and its independent GC audit."""
from gc_common import *


def main():
    profile, gc = read(BASE / 'model-analysis.json'), read(GC_BASE / 'analysis.json')
    closures = []
    for folder, filename, analysis_name in [(BASE, 'model-closed.json', 'model-analysis.json'), (GC_BASE, 'closed.json', 'analysis.json')]:
        proof = read(folder / filename)
        assert proof['passed'] and proof['analysis'] == pin(folder / analysis_name)
        verify_spec(proof)
        for identity in proof['identities']:
            terminal(identity)
        closures.append(pin(folder / filename))
    report, observations = TOOLS / 'results-20260922.md', TOOLS / 'observations-20260922.json'
    assert not report.exists() and not observations.exists()
    save(observations, dict(profile_closure=closures[0], gc_closure=closures[1], core=CORE, data=DATA,
        profile=profile, gc={**gc, 'captures': [{key: value for key, value in c.items() if key not in ['pauses', 'collections']} for c in gc['captures']]}))
    labels = [('Packed three-row matrix kernel', 'MathOps.mm_unsafe_vectorized_intrinsics_3x4packed('),
        ('Tiled convolution caller', '.RunTiledBatchFloat('), ('Memmove', '.SpanHelpers.Memmove('),
        ('LSTM provider', '.CPUExecutionProvider.Lstm('), ('Ordered LSTM projection', '.LstmProjectOrdered('),
        ('Packed two-row remainder kernel', '.mm_unsafe_vectorized_intrinsics_2x4packed_bump('),
        ('Log-mel caller', '.WeSpeakerAudio.LogMelFilterbank('), ('Fourier transform', '.WeSpeakerAudio.Fourier(')]
    rows = []
    for label, pattern in labels:
        values = []
        for capture in profile['diagnostics']:
            seconds = sum(row['seconds'] for row in capture['exclusive'] if row['marker'] == 'dialogue-30s' and pattern in row['method'])
            values.append((seconds, seconds / capture['selected_seconds']['dialogue-30s']))
        rows.append(f'| {label} | {values[0][0]:.6f} | {values[0][1]:.2%} | {values[1][0]:.6f} | {values[1][1]:.2%} |')
    timings = '\n'.join(f"| {r['name']} | {r['roles']['control']['wall_mean']:.6f} | {r['roles']['sampled-a']['wall_mean']:.6f} | {r['roles']['sampled-b']['wall_mean']:.6f} |"
        for r in profile['observations'])
    gc_rows = '\n'.join(f"| {capture['name']} | {r['name']} | {r['gc_envelope_ms']:.3f} | {r['gc_envelope_to_wall']:.3%} |"
        for capture in gc['captures'] for r in capture['totals'])
    report.write_text(f'''# Complete-request profile of the integrated pyannote candidate

The current three-row matrix kernel dominates both captures: **53.74% and
54.34%** of selected full-request managed thread time. The tiled convolution
caller accounts for another **15.53% and 15.30%**. All **48 public requests**
pass the original native, input, repeat and held-output checks, preserving the
qualified integrated build's outputs exactly.

| Exclusive leaf | A seconds | A share | B seconds | B share |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

Each capture includes three full requests and three calls to each crop. The
shares use exclusive leaves, which reconcile with the selected total; inclusive
callers overlap. Inlining can charge helper work to a caller. In particular,
15% in RunTiledBatchFloat does not isolate its bias loop or tensor constructors.
These are sampled managed thread-time weights, not native/kernel CPU samples.
The exporter's synthetic `CPU_TIME` bucket retains its original label without
being interpreted as measured CPU usage. Process CPU is reported separately.

## Complete coverage and diagnostic variation

| Fixture | Control mean seconds | Capture A mean | Capture B mean |
|---|---:|---:|---:|
{timings}

The two exports agree on every stack event. Each fixture has all three marker
intervals on the correct target thread and passes the original coverage limits.
Full-request sampled/wall totals are 34.100872/34.157162 seconds and
34.857568/34.882018 seconds. Corresponding process CPU totals are 34.000000 and
33.546875 seconds. All fixture coverage and every leaf are retained in the
observations. Control/capture differences combine diagnostic overhead and
process variation; they do not establish a new speedup or ORT ratio.

The consumer source differs from the previously verified diagnostic only in
its two required Core/Data digest literals. All barriers, wrappers, counters,
validation and ownership checks remain identical. Product binaries are not
rebuilt. Targets use Windows i7-14700KF CPU2, collectors CPU0, normal .NET10.0.12,
four warmups and three measured passes per fixture. The pinned collector is
dotnet-trace10.0.745401 with the original profiles and diagnostic EventSource.

Core: `{CORE}`.
Data: `{DATA}`.

## GC attribution from the same captures

The exact previously qualified offline reader and 17-tested parser process
these new traces without another model run. All 24 captured requests reconcile
108 collection starts with their original generation counters; reported event
loss is zero. Background collections crossing requests remain explicit.

| Capture | Fixture | GC suspension envelope ms, three calls | Envelope / wall |
|---|---|---:|---:|
{gc_rows}

GC suspension is an upper envelope of **1.63% and 1.29%** of full-request wall
time. The [GC analysis methodology](../retained-gc/results-20260922.md) pairs
suspension phases by emitting thread, clips interval unions to each request,
and separates GC, GC preparation and other reasons. This does not measure
background GC CPU or identify the cause of unprofiled timing variation.

## Next bounded experiment

RunTiledBatchFloat currently creates three DenseTensor views and their shape
metadata before trying TryConvPortableRows. That helper consumes only spans.
Its fallback needs the views, while its successful path does not. Test moving
their construction into the fallback and supplying the same sliced spans to
the fast helper. This preserves the existing admission predicate, matrix
arithmetic, packing, destination clearing, bias order, storage and scalar path.
The primary AVX-512 candidate already places its own view construction after
its specialized helper refuses, but no combined dispatch is selected here.

The complete static census has 6,147 portable-eligible tile products per
embedding call under the stated default FMA policy, or 18,441 transient views.
A full 30-second request has 21 embedding windows. These are source/shape counts,
not measured eliminated allocations or a speedup. Qualify the one-method change,
measure actual graph/request allocations, preserve complete outputs and require
a fresh controlled application comparison before any performance admission.
The matrix kernel remains the largest computation target; this small caller
experiment does not replace the queued AVX-512 target evaluation.

## Evidence and reproduction

Profile preparation session6537 and execution18732 exit zero before audit.
All **{profile['resource_samples']} profile resource samples** and **{len(profile['identities'])} process identities** pass;
the independent GC phase adds **{gc['resource_samples']} resource samples** and **{len(gc['identities'])} terminal identities**.
Peak sampled profile RSS is **{max(r['peak_rss'] for r in profile['resources']):,} bytes**.
Original bounds remain 10 GiB capture preflight, 8 GiB aggregate RSS, 1 GiB
available, 20 GiB disk, 1 GiB output and 900 seconds per pair. Builds/stack
exports use 8 GiB preflight; offline GC reads use 2 GiB preflight/RSS and 300s.

Profile closure: `{closures[0]['sha256']}` ({closures[0]['bytes']:,} bytes).
GC closure: `{closures[1]['sha256']}` ({closures[1]['bytes']:,} bytes).
[Full observations](observations-20260922.json) retain all fixtures and summary
inputs; raw traces, exports and complete GC records remain in the closed
artifact directories. Use `C:/Python313/python.exe -X utf8 -B` with prepare.py,
run.py, audit.py, gc.py, audit_gc.py and report.py, waiting for each actual exit.
Existing outputs are refused. Production sources, accepted benchmark tables,
live e5 and the queued primary AMD payload are unchanged.
''', encoding='utf8')
    print(json.dumps(dict(report=rel(report), profile=closures[0], gc=closures[1])))


if __name__ == '__main__':
    main()
