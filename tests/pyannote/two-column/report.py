"""Report all fixed gates without converting component gains into model claims."""
from pathlib import Path
import run

c=run.common


def main():
    closed=c.read(c.BASE / 'closed.json'); assert closed['passed']; c.verify(closed['files'])
    a=c.read(c.BASE / 'analysis.json'); spec=c.read(c.BASE / 'payload/payload.json')
    assert a['passed'] and a['validation_cases']==3266
    result=run.TOOLS / 'results-20260922.md'; observations=run.TOOLS / 'observations-20260922.json'
    assert not result.exists() and not observations.exists()
    table=['| Rows / reduction / columns | Baseline µs | Candidate µs | Candidate / baseline | Baseline repeat | Candidate repeat |',
        '|---|---:|---:|---:|---:|---:|']
    for r in a['rows']:
        m=r['process_means']; b=(m['baseline-a']+m['baseline-b'])/2; candidate=(m['candidate-a']+m['candidate-b'])/2
        table.append(f"| {r['m']} / {r['n']} / {r['k']} | {b*1e6:.3f} | {candidate*1e6:.3f} | {r['candidate_baseline']:.6f} | {r['controls']['baseline']:.6f} | {r['controls']['candidate']:.6f} |")
    verdict='ELIGIBLE FOR FULL-MODEL QUALIFICATION' if a['eligible'] else 'NOT SELECTED'
    samples=sum(r['samples'] for r in a['resources']); peak=max(r['peak_rss'] for r in a['resources'])
    text=f'''# Separate two-column convolution routine on AMD

Fixed component verdict: **{verdict}**. All correctness and resource checks pass.
Equal-shape geometric mean candidate/baseline is **{a['geomean_candidate_baseline']:.6f}**
(required <=0.95); worst shape is **{a['max_shape_candidate_baseline']:.6f}**
(required <=1.05). Every repeated-role shape control passes:
**{a['controls_passed']}** (each <=1.10). These gates were fixed before execution.

{chr(10).join(table)}

Means include clearing, packing, multiplication and bias/copy to actual final
row strides and last tile offsets. Buffers are preallocated; patch expansion,
pool rental and tensor-view construction are outside this synthetic tile scope.
The baseline uses exact selected raw kernels and the original scalar pointer
epilogue. Equal-shape weighting is not application-frequency weighting.

Four fresh normal AMD processes run baseline/candidate/candidate/baseline.
One second of conditioning for every shape precedes a further per-shape
one-second warmup and six measured blocks. All **528 timing blocks**, all22
shapes and all44controls are retained. Blocks within a process are not
independent process replications or confidence bounds.

## Mechanism and qualification

The [AMD code-generation diagnostic](../tail-codegen/results-20260922.md)
found repeated stack address reloads inside the rejected direct-output narrow
loops. This successor derives from the masked-store v4 candidate and isolates
exactly two columns in a private NoInlining routine. It advances three input-row
pointers and the packed B pointer during reduction and derives output addresses
afterward. Masked operations, 256-bit lanes, separate multiply/add recurrence,
bias NaN selection, admission and the two-row remainder are unchanged.
All other paths remain v4. This timing screen does not itself demonstrate the
new helper's emitted instruction sequence or attribute cycles to reloads.

The original consumer source and full expanded manifest are unchanged.
Normal and forced-scalar validation pass on Windows and AMD: **3,266 cases
per mode**, **{a['validation_values']:,} active values**, **{a['checked_buffer_values']:,}
complete buffer positions**, including **{a['nonfinite_values']:,} NaN outputs**
per mode. All candidate/baseline bits, input ownership and guards pass. The
independent scalar oracle checks finite/infinity bits and NaN classification;
finite/zero hashes agree across the four combinations. NaN payloads remain
exact against the selected baseline for each mode. Forced-scalar mode disables
both tail predicates in its separate assembly and exercises the original scalar
fallback with normal FMA hardware; it is not physical AVX2-off execution.

AMD EPYC9V74 uses normal .NET10.0.8 without diagnostic or ISA flags. Target
CPU2 affinity is inherited before CLR, and every observed native thread stays
onCPU2; monitorCPU0. All **{samples:,} resource observations** pass; peak
aggregate worker RSS is **{peak:,} bytes**. Preparation54300, supervisor
670928/birth1790052330.58 and all six targets terminate with exit zero.
Collection and independent audit verify every input hash, resource sample,
output digest and original gate. No failed sample or shape is discarded.

Selected Core: {c.CORE}.
Normal probe: {spec['consumer']['sha256']}.
Forced-scalar probe: {spec['scalar_consumer']['sha256']}.
Closure: {c.pin(c.BASE / 'closed.json')['sha256']}.
Artifacts: artifacts/pyannote-two-column-20260922.

Root product source is unchanged. Accepted full-dialogue latency remains
**Lokad.Onnx15.466s versus Microsoft ORT8.952s**, ratio **1.728**. This component
screen supplies no new application speedup or ORT ratio.
'''
    if a['eligible']:
        text+='\nNext: compose the convolution-only candidate in an isolated normal build,\nverify unchanged other methods/public declarations, then qualify complete\ngraphs, public requests, meetings, shared models, suites and package consumers\nbefore a fresh matched AMD production/candidate/ORT application comparison.\n'
    else:
        text+='\nDo not integrate this candidate or repeat the unchanged screen. Preserve all\nfailed gates and select a distinct next mechanism from the evidence.\n'
    c.save(observations,dict(**a,closure=c.pin(c.BASE / 'closed.json'),analysis=c.pin(c.BASE / 'analysis.json')))
    result.write_text(text,encoding='utf8')
    print(dict(verdict=verdict,report=c.pin(result)))


if __name__=='__main__':main()
