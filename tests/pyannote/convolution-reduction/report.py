"""Report both fixed gates and all shapes after the independent AMD audit."""
import common
common.BASE = common.ROOT / 'artifacts/pyannote-convolution-reduction-v2-20260922'
from common import *


def main():
    closed=read(BASE / 'closed.json'); assert closed['passed']; verify(closed['files'])
    a=read(BASE / 'analysis.json'); assert a['passed']
    output=TOOLS / 'results-20260922.md'; observations=TOOLS / 'observations-20260922.json'
    assert not output.exists() and not observations.exists()
    table=['| Rows / reduction / columns | Baseline µs | Candidate µs | Candidate / baseline | Baseline repeat ratio | Candidate repeat ratio |',
        '|---|---:|---:|---:|---:|---:|']
    for r in a['rows']:
        m=r['process_means']; baseline=(m['baseline-a']+m['baseline-b'])/2; candidate=(m['candidate-a']+m['candidate-b'])/2
        table.append(f"| {r['m']} / {r['n']} / {r['k']} | {baseline*1e6:.3f} | {candidate*1e6:.3f} | {r['candidate_baseline']:.6f} | {r['controls']['baseline']:.6f} | {r['controls']['candidate']:.6f} |")
    verdict='ELIGIBLE FOR MODEL QUALIFICATION' if a['eligible'] else 'NOT SELECTED'
    resources=sum(r['samples'] for r in a['resources']); peak=max(r['peak_rss'] for r in a['resources'])
    text=f'''# Three-row convolution reduction panels on AMD

Fixed component verdict: **{verdict}**. Correctness and resource checks pass.
The geometric mean candidate/baseline ratio is **{a['geomean_candidate_baseline']:.6f}**
(required at most 0.95); the largest shape ratio is
**{a['max_shape_candidate_baseline']:.6f}** (required at most 1.05).
All same-role process controls pass: **{a['controls_passed']}** (each at most 1.10).
These gates were frozen before timing and are not adjusted for the outcome.

{chr(10).join(table)}

Means include destination clearing, packing and multiplication. The 22 actual
convolution tile geometries have equal weight for this component screen;
their geometric mean is not a complete-request speedup. Six measured blocks
in each of four fresh processes give {a['measured_blocks']} retained blocks.
Every process conditions every shape for at least one second, then warms each
shape for at least one second before measurement. The fixed order is baseline,
candidate, candidate, baseline. All observations are retained; blocks inside
one process are not independent process replications or confidence intervals.

Both Windows and AMD pass **{a['validation_cases']:,} guarded arithmetic cases**,
covering **{a['validation_values']:,} output values** per platform. Every destination
bit matches the selected kernel and independent scalar FMA/multiply-add oracle,
including nonzero destinations, offsets, all guards and input preservation.
The two platforms produce identical validation records. All timed outputs
match the validated complete-buffer hashes, with unchanged inputs.

The candidate is generated from the current three-row packed method. Only
the full 32-column row loop is wrapped in ascending 128-term reduction blocks,
and only reductions above 128 enter it. Every float32 accumulator resumes with
the next original term. Existing vector/narrow tail source, short-reduction
route and two-row remainder remain unchanged. Product DLL bytes are unchanged.
The selected Core is e9c87932b2184c2f6bfef72faabb1719bdbceadc779a15fe1ffd3f3056d02838;
consumer is c5ea607d75e541193c87e99df760e86dd540fad0919c26311a2be7cfdda5d4cb.

AMD EPYC 9V74 uses normal .NET 10.0.8 with no runtime flags, CPU 2 inherited
before CLR, and monitor CPU 0. Every observed native thread affinity passes.
All **{resources:,} resource observations** pass, with peak aggregate RSS
**{peak:,} bytes** across local preparation and AMD execution. Supervisor
666152 / birth 1790047881.64 and all five remote targets are terminal with
exit zero. Collection verifies unchanged frozen inputs; independent audit
checks all resources, output hashes, repetitions and original gates.

The [executed protocol](execution.md) preserves the initial encoding-name
failure before builds and its additive correction. Local build has zero
warnings and zero errors. Six payload and 222 remote runtime files were pinned.
All raw arrays-by-hash, timing blocks, source, generated kernel, executable,
logs and process/resource evidence are retained under
artifacts/pyannote-convolution-reduction-v2-20260922.
Closure: {pin(BASE / 'closed.json')['sha256']}.

The [older e5 reduction-blocking experiment](../../e5/projection-reduction-timing/results-20260920.md)
remains rejected. It measured a different 12/8-row AVX-512 prepared projection
workload. Neither it nor this synthetic tile experiment measures native ORT
or the full diarization request. The accepted application comparison remains
15.466 seconds versus Microsoft ORT 8.952 seconds; root production is unchanged.
'''
    if a['eligible']:
        text+='\nNext: build an isolated convolution-only product candidate, prove unchanged\nshared dispatch/public surface, then qualify complete graphs and requests\nbefore a fresh matched AMD application comparison. No promotion follows yet.\n'
    else:
        text+='\nDo not integrate this candidate or repeat the unchanged screen. Preserve all\nfailed gates and use a distinct mechanism for further Pyannote optimization.\n'
    save(observations,dict(**a,closure=pin(BASE / 'closed.json'),analysis=pin(BASE / 'analysis.json')))
    output.write_text(text,encoding='utf8')
    print(dict(verdict=verdict,report=pin(output),observations=pin(observations)))


if __name__=='__main__': main()
