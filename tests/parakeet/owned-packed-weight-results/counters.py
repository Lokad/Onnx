"""Publish measured corpus accounting, distinct from application timing."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'owned-packed-weight-counters-resume'))
from run import BASE,prepared,pin,read


def main():
    assert not (HERE/'counters-20260925.json').exists() and not (HERE/'counters-20260925.md').exists()
    prepared();closed=read(BASE/'closed.json');analysis=read(BASE/'analysis.json')
    assert closed['passed'] and closed['analysis']==pin(BASE/'analysis.json') and closed['reviewer']==pin(BASE.parents[1]/'tests/parakeet/owned-packed-weight-counters-resume/review.py')
    for name,wanted in closed['files'].items():assert pin(BASE/name)==wanted,name
    assert analysis['passed'] and not analysis['application_scored'] and not analysis['release_admitted'] and not analysis['completed_control_repeated']
    assert [r['mode'] for r in analysis['comparisons']]==['512','256']
    for row in analysis['comparisons']:
        assert row['avoided_packs']==1740 and row['reconstructions']==609
        assert row['scratch_reduction']==29192355840 and row['copy_increase']==10217324544
        assert len(row['clips'])==20 and all(c['outputs_exact'] for c in row['clips'])
    report=dict(passed=True,diagnostic_only=True,performance_measured=False,release_admitted=False,
        closure=pin(BASE/'closed.json'),analysis=pin(BASE/'analysis.json'),products=analysis['products'],
        model_closure=analysis['model_closure'],census_closure=analysis['census_closure'],comparisons=analysis['comparisons'],
        resources=analysis['resources'],initial_failure=analysis['initial'],completed_control_repeated=False,
        failed_release_controls=analysis['failed_release_controls'],publisher=pin(Path(__file__)))
    (HERE/'counters-20260925.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    prose='''# Actual Parakeet packing and reconstruction counts

The corrected M76 candidate takes the intended path on all20 public-corpus
encoder inputs in both instruction modes. For each corpus it avoids1,740
transient16MiB packs (29,192,355,840 cumulative scratch bytes). Seven sequence
lengths need87 dense weight reconstructions each:609 copies,10,217,324,544
cumulative copy bytes. The other13 clips add no weight-copy bytes. Every
per-clip difference matches the prospective prediction exactly.

The affected lengths are167,89,157,83,61,151 and169: odd values not divisible
by three. Existing multiplication and remainder arithmetic is unchanged.
The candidate removes per-call packing for87 owned constants while retaining
the nine already prepared feed-forward constants and all37 existing cache
entries. Profiler CopyY stages identify every reconstruction at its actual
node; none appears on the other owned calls or untouched prepared weights.

Across80 encoder requests and7,680 feed-forward calls, frontend and encoder
outputs are hash-identical between selectedM73 and correctedM76. Audio/features
remain immutable, held outputs remain independent, and actual shapes, runtime
identities, instruction modes, affinity and resource checks pass. Core and Data
are the same binaries used for full native/public model qualification. Only the
diagnostic consumer was built, and profiler durations are discarded.

The first selected-512 worker completed before an immediate memory preflight
refused to start candidate-512. Its complete original result remains retained.
Recovery reuses the same consumer, waits for the unchanged11GiB guard, and runs
only the three unstarted workers. The original code1 remains recorded; no
successful worker is repeated. The earlier compiler-warning record is also
retained with its explicit Linux-guard correction.

These are cumulative accounting totals, not peak memory, hardware-bandwidth
measurements or application gains. The next step is the matched complete
application comparison against fresh ORT timings, with all existing controls
and the3% gain criterion. Prior release failures still prevent promotion;
BENCHMARK.md remains on the qualified release.

Full per-clip counts, product hashes, resources and failure provenance are in
[counters-20260925.json](counters-20260925.json). Raw collections remain under
artifacts/parakeet-owned-packed-weight-counters-v2-amd-20260925 and
artifacts/parakeet-owned-packed-weight-counters-resume-amd-20260925.
'''
    (HERE/'counters-20260925.md').write_text(prose,encoding='utf8')
    print(json.dumps(dict(report=pin(HERE/'counters-20260925.json'),closure=pin(BASE/'closed.json'),avoided_packs_per_corpus=1740,reconstructions_per_corpus=609)))


if __name__=='__main__':main()
