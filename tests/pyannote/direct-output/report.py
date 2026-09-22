"""Publish the direct-output component verdict only after independent closure."""
import complete_v4
from common import *


def main():
    closed=read(BASE / 'closed.json'); assert closed['passed']; verify(closed['files'])
    a=read(BASE / 'analysis.json'); assert a['passed']
    spec=read(BASE / 'payload/payload.json')
    output=TOOLS / 'results-20260922.md'; observations=TOOLS / 'observations-20260922.json'
    assert not output.exists() and not observations.exists()
    verdict='ELIGIBLE FOR FULL-MODEL QUALIFICATION' if a['eligible'] else 'NOT SELECTED'
    table=['| Rows / reduction / columns | Final row stride | Baseline µs | Direct-output µs | Candidate / baseline | Baseline repeat | Candidate repeat |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in a['rows']:
        m=r['process_means']; baseline=(m['baseline-a']+m['baseline-b'])/2; candidate=(m['candidate-a']+m['candidate-b'])/2
        table.append(f"| {r['m']} / {r['n']} / {r['k']} | {r['stride']} | {baseline*1e6:.3f} | {candidate*1e6:.3f} | {r['candidate_baseline']:.6f} | {r['controls']['baseline']:.6f} | {r['controls']['candidate']:.6f} |")
    resources=sum(r['samples'] for r in a['resources']); peak=max(r['peak_rss'] for r in a['resources'])
    text=f'''# Direct final-row convolution output on AMD

Fixed component verdict: **{verdict}**. All correctness and resource checks pass.
The equal-shape geometric mean candidate/baseline ratio is
**{a['geomean_candidate_baseline']:.6f}** (required <=0.95); the largest shape ratio
is **{a['max_shape_candidate_baseline']:.6f}** (required <=1.05).
Every repeated-role shape control passes: **{a['controls_passed']}** (each <=1.10).
These gates were fixed before timing; no sample or failed control is removed.

{chr(10).join(table)}

The 22 shapes use their actual final row strides and last actual tile offsets,
with bias as present in all 36 embedding convolutions. Means include temporary
clearing, packing, multiplication and bias/copy. Buffers are preallocated;
patch expansion, pool rental and tensor view construction are outside this
synthetic tile scope. The baseline uses the exact selected raw kernels and a
scalar pointer epilogue implementing the original bias/copy operations. It is
not the complete production caller or a public diarization request.

Four fresh normal AMD processes run baseline/candidate/candidate/baseline.
Each first conditions all shapes for at least one second each, then warms
each shape again before six measured blocks. All **{a['measured_blocks']} blocks**
are retained. Equal-shape weighting is a component screen, not application
frequency weighting; repeated blocks are not independent process replications
or a confidence interval. The original full-application ORT table is unchanged.

## Correctness and mechanism

Normal and forced-scalar validation pass on both Windows and AMD: **{a['validation_cases']:,}
cases per mode**, checking **{a['validation_values']:,} active output values** and
**{a['checked_buffer_values']:,} complete buffer positions** per mode. Cases cover
all 22 real geometries and 33 first/last offsets, reduction and SIMD boundaries,
empty inputs, with/without bias, signed zeros and non-finite values.
Each mode includes {a['nonfinite_values']:,} NaN outputs. Candidate/baseline bits,
padding, guards and input preservation are exact. The independent scalar oracle
checks finite/infinity bits and NaN classification. Finite/zero complete-buffer
hashes agree across all four platform/mode combinations. NaN payloads retain
exact per-mode baseline comparisons without inventing a cross-mode contract.

The candidate keeps the selected portable admission and row assignment. It
starts three-row accumulators at positive zero, preserves every ascending FMA
or separate multiply/add reduction, applies bias afterward and writes to the
final row stride. The remaining two/four rows use the unchanged two-row kernel
and scalar epilogue. Non-admitted shapes retain the original route. No reduction
blocking or shared MatMul dispatcher change is included.

The [executed successor](execution-v4.md) retains three earlier failures:
AVX2 disabling also disabled FMA before arithmetic; forced-scalar Windows
qualification exposed a bias NaN-payload mismatch; AMD exposed that mismatch
in its SIMD path. The corrected generator makes bias NaN selection explicit
in both paths. Every original case and timing gate remains unchanged, and no
timing process ran in a failed preparation/qualification attempt.
The separate scalar-tail validation assembly changes only the tail predicates
of the generated candidate and original 3/2-row copies, plus four consumer call
targets. It executes with normal FMA-capable hardware and no runtime flags;
it does not claim physical FMA-on/AVX2-off execution.

Selected Core: {CORE}.
Normal probe: {spec['consumer']['sha256']}.
Scalar-tail probe: {spec['scalar_consumer']['sha256']}.
Root production and its DLL bytes are unchanged.

## Execution and evidence

AMD EPYC 9V74 uses normal .NET 10.0.8, target CPU 2 inherited before CLR and
monitor CPU 0. Every observed native thread affinity passes. All **{resources:,}
resource observations** pass, with peak aggregate worker RSS **{peak:,} bytes**
across local preparation and AMD execution. All local owners, remote supervisor
668034 / birth 1790049774.53 and six targets are terminal with successful exits.
Collection and independent audit verify frozen files, all output hashes,
resource bounds, complete repetitions and every original timing gate.

Artifacts: artifacts/pyannote-direct-output-v4-20260922.
Closure: {pin(BASE / 'closed.json')['sha256']}.
The v3 AMD failure remains separately closed at
c07efe05c37a1a30653d2a2c3f9fe7d1a82d255138f1b80c2a3d516f5d66c997.

The accepted full application comparison remains 15.466 seconds for Lokad.Onnx
versus Microsoft ORT 8.952 seconds (1.728 ratio). This experiment establishes
no new full-request speedup, ORT ratio or production promotion.
'''
    if a['eligible']:
        text+='\nNext: build the isolated convolution-only product candidate, verify all other\ncompiled methods/public declarations, then qualify full graphs, requests and\nmeetings before a fresh matched AMD application comparison.\n'
    else:
        text+='\nDo not integrate this candidate or repeat the unchanged screen. Retain the\nfailed gates and choose a distinct mechanism for further optimization.\n'
    save(observations,dict(**a,closure=pin(BASE / 'closed.json'),analysis=pin(BASE / 'analysis.json')))
    output.write_text(text,encoding='utf8')
    print(dict(verdict=verdict,report=pin(output),observations=pin(observations)))


if __name__=='__main__': main()
