"""Report every fixed gate after complete store-width and AMD qualification."""
from pathlib import Path
import complete
from common import *

OUTPUT=Path(__file__).resolve().parent


def main():
    closed=read(BASE / 'closed.json'); assert closed['passed']; verify(closed['files'])
    a=read(BASE / 'analysis.json'); assert a['passed'] and a['validation_cases']==3266
    spec=read(BASE / 'payload/payload.json')
    result=OUTPUT / 'results-20260922.md'; observations=OUTPUT / 'observations-20260922.json'
    assert not result.exists() and not observations.exists()
    table=['| Rows / reduction / columns | Baseline µs | Ordinary-store µs | Candidate / baseline | Baseline repeat | Candidate repeat |',
        '|---|---:|---:|---:|---:|---:|']
    for r in a['rows']:
        m=r['process_means']; b=(m['baseline-a']+m['baseline-b'])/2; c=(m['candidate-a']+m['candidate-b'])/2
        table.append(f"| {r['m']} / {r['n']} / {r['k']} | {b*1e6:.3f} | {c*1e6:.3f} | {r['candidate_baseline']:.6f} | {r['controls']['baseline']:.6f} | {r['controls']['candidate']:.6f} |")
    verdict='ELIGIBLE FOR FULL-MODEL QUALIFICATION' if a['eligible'] else 'NOT SELECTED'
    samples=sum(r['samples'] for r in a['resources']); peak=max(r['peak_rss'] for r in a['resources'])
    text=f'''# Ordinary narrow stores in direct-output convolution

Fixed component verdict: **{verdict}**. Correctness and resource checks pass.
The equal-shape geometric mean candidate/baseline ratio is
**{a['geomean_candidate_baseline']:.6f}** (required <=0.95); the largest shape ratio
is **{a['max_shape_candidate_baseline']:.6f}** (required <=1.05).
Every repeated-role shape control passes: **{a['controls_passed']}** (each <=1.10).
The complete 22-geometry workload and every original gate are retained.

{chr(10).join(table)}

Means include clearing, packing, multiplication and bias/copy to each shape's
actual final row stride and last tile offset. Buffers are preallocated; patch
expansion, pool rental and tensor-view construction are outside this synthetic
tile scope. Equal-shape weighting is not application frequency weighting.
The baseline retains the exact selected raw kernels and scalar pointer epilogue.
Neither side is the complete public diarization request.

Four fresh normal AMD processes run baseline/candidate/candidate/baseline.
All shapes receive one second of conditioning and a further one-second warmup
before six measured blocks. All **528 timing blocks** are retained. Blocks
inside a process are not independent process replications or confidence bounds.

## Changed mechanism and complete coverage

The [previous direct-output candidate](../direct-output/results-20260922.md)
remains rejected: its equal-shape mean improved 6.73%, but two-column tails
regressed 12.69% and 5.35%. This successor changes only the three masked final
stores and their unused integer casts, adding a private helper that writes
exactly one-to-seven valid floats with ordinary 16-byte, 8-byte and scalar
stores. The eight-byte operations copy bits; they do not perform double
arithmetic. Masked input loads, ascending reductions, explicit bias NaN rules,
admission, final output stride and the original two-row remainder are unchanged.
No shape is disabled to conceal the previous regression.

The original 2,882 qualification cases are retained as an exact prefix. Review
before VM deployment added 384 cases to exercise all new store widths three
through six, both alone and after a 32-column panel. The consumer source and
all timing-shape records are identical to the previous experiment. Both normal
and forced-scalar modes pass on Windows and AMD: **3,266 cases per mode**,
**{a['validation_values']:,} active output values** and **{a['checked_buffer_values']:,}
complete buffer positions** per mode. Each mode includes {a['nonfinite_values']:,}
NaN outputs. All candidate/baseline bits, padding, guards and input checks pass.
The independent scalar oracle retains finite/infinity bits and NaN class;
finite/zero full-buffer hashes agree across all four platform/mode combinations.
NaN payloads remain exact against each mode's selected baseline.

The [executed protocol](execution.md) preserves the initial partial-width
preparation and its complete successor. The initial preparation never reached
the VM. Earlier direct-output numerical failures remain in their original
artifacts, with the explicit bias NaN correction unchanged. No numerical,
resource or timing threshold is relaxed.

## Runtime and ownership

AMD EPYC 9V74 uses normal .NET 10.0.8, CPU 2 inherited before CLR and CPU 0
for the monitor. Every observed native thread affinity passes. All **{samples:,}
resource observations** pass; peak aggregate worker RSS is **{peak:,} bytes**
across local preparation and AMD execution. All local owners, supervisor
669088 / birth 1790050620.12 and six remote targets are terminal with exit zero.
Collection and independent audit verify frozen inputs, complete observations,
output hashes and every original timing gate.

Selected Core: {CORE}.
Normal probe: {spec['consumer']['sha256']}.
Forced-scalar probe: {spec['scalar_consumer']['sha256']}.
Closure: {pin(BASE / 'closed.json')['sha256']}.
Artifacts: artifacts/pyannote-direct-output-store-v2-20260922.

Root product source remains unchanged. The accepted application comparison is
still Lokad.Onnx 15.466 seconds versus Microsoft ORT 8.952 seconds (1.728 ratio).
This component result establishes no new full-request speedup or ORT ratio.
'''
    if a['eligible']:
        text+='\nNext: adapt the proved kernel into an isolated convolution-only product build,\n+prove unchanged other methods/public declarations, then complete graph/public/\n+meeting/shared/suite/package qualification and a fresh matched AMD comparison.\n'.replace('\n+','\n')
    else:
        text+='\nDo not integrate this candidate or repeat the unchanged screen. Preserve every\nfailed gate and select a distinct next mechanism from the evidence.\n'
    save(observations,dict(**a,closure=pin(BASE / 'closed.json'),analysis=pin(BASE / 'analysis.json')))
    result.write_text(text,encoding='utf8')
    print(dict(verdict=verdict,report=pin(result),observations=pin(observations)))


if __name__=='__main__': main()
