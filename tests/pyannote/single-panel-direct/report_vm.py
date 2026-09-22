"""Publish the completed component screen and retain every raw timing block."""
import json
from vm import TOOLS, BASE, LOCAL, prepared, pin, read, verify


def main():
    prepared()
    closed = read(BASE / 'closed.json'); assert closed['passed']; verify(closed['files'])
    analysis = read(BASE / 'analysis.json'); assert analysis['passed']
    assert closed['analysis'] == pin(BASE / 'analysis.json')
    report = TOOLS / 'results-amd-20260922.md'; observations = TOOLS / 'observations-amd-20260922.json'
    assert not report.exists() and not observations.exists()
    raw = {name: read(BASE / 'collected/output' / (name + '.json'))
           for name in ['baseline-a', 'candidate-a', 'candidate-b', 'baseline-b']}
    assert sum(len(r['records']) for r in raw.values()) == 528
    observations.write_text(json.dumps(dict(closure=pin(BASE / 'closed.json'), raw=raw, **analysis), indent=2) + '\n', encoding='utf8')
    verdict = 'passes' if analysis['eligible'] else 'fails'
    rows = '\n'.join(f"| {r['m']} / {r['n']} / {r['k']} | {r['candidate_baseline']:.6f} | {r['controls']['baseline']:.6f} | {r['controls']['candidate']:.6f} |" for r in analysis['rows'])
    report.write_text(f'''# Single-panel direct-output component on AMD

The new component **{verdict} the original eligibility gates**. All correctness
and resource checks pass. Equal-shape geometric mean candidate / production is
**{analysis['geomean_candidate_baseline']:.6f}**, against the fixed 0.95 limit;
the worst shape is **{analysis['max_shape_candidate_baseline']:.6f}**, limit 1.05.
All 44 repeated-process controls pass: **{analysis['controls_passed']}**.

The candidate combines direct final-output writes with omission of the packing
copy for admitted patches of at most 32 columns. Production retains its ordinary
packed three-row/two-row kernels and bias/copy epilogue. These component timings
include clearing, packing when needed, multiplication and bias/copy. They exclude
patch formation, scratch rental and tensor views. The comparison does not isolate
the causal benefit of copy removal, and supplies no application or ORT ratio.

All 3,266 cases pass in each Windows/AMD normal/scalar-tail mode. The
[independent local layout proof](results-local-20260922.md) additionally passes
1,400 cases per normal/hardware-disabled mode. Arithmetic bodies, validators,
all 22 actual shapes and their output strides remain unchanged.

AMD EPYC 9V74, CPU2, .NET 10.0.8, accepted Core `e9c87932`. Four fresh timing
processes run baseline, candidate, candidate, baseline, retaining six measured
blocks per shape/process after the original conditioning. All 528 raw blocks
and conditioning records are retained in [observations](observations-amd-20260922.json).
No sample exclusion or unchanged timing retry is permitted.

| Rows / reduction / columns | Candidate / production | Baseline max/min | Candidate max/min |
|---|---:|---:|---:|
{rows}

All {sum(r['samples'] for r in analysis['resources'])} AMD resource samples pass;
peak owned RSS is {max(r['peak_rss'] for r in analysis['resources']):,} bytes.
All seven remote supervisor/target identities are terminal before collection.
The local proof separately retains 123 resource observations and 17 terminal
identities. Original CPU, runtime, memory, tmpfs, artifact and time bounds hold.

The [preceding full application trial](../direct-amd-results/results-20260922.md)
remains unselected at ratio 0.970383 against its 0.970000 gate. This component
does not amend that result. Normal product composition, scratch-accounting
checks, full model/public/meeting qualification and fresh application timing
are required before any root integration. Production remains unchanged.

Artifact: artifacts/pyannote-single-panel-amd-20260922.
Closure: {pin(BASE / 'closed.json')['sha256']}.
Local closure: {pin(LOCAL / 'closed.json')['sha256']}.
Reproduction is vm.py prepare/stage/launch/observe/collect, then audit_vm.py and
this report after terminal owners. Existing output directories are not restart targets.
''', encoding='utf8')
    print(json.dumps(dict(eligible=analysis['eligible'], controls=analysis['controls_passed'],
        geomean=analysis['geomean_candidate_baseline'], worst=analysis['max_shape_candidate_baseline'],
        report=str(report), closure=pin(BASE / 'closed.json'))))


if __name__ == '__main__': main()
