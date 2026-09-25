"""Publish actual packing/copy traffic without presenting it as latency savings."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'packed-final-row-counters'))
from run import BASE,prepared,pin,read,write


def main():
    paths=[HERE/('counters-20260925'+s) for s in ['.json','.md']]
    assert not any(p.exists() for p in paths)
    prepared();closed=read(BASE/'closed.json');analysis=read(BASE/'analysis.json')
    assert closed['passed'] and analysis['passed'] and closed['analysis']==pin(BASE/'analysis.json')
    assert closed['reviewer']==pin(HERE.parent/'packed-final-row-counters/review.py')
    for name,wanted in closed['files'].items():assert pin(BASE/name)==wanted,name
    assert not analysis['application_scored'] and not analysis['release_admitted']
    assert [r['mode'] for r in analysis['comparisons']]==['512','256']
    for row in analysis['comparisons']:
        assert row['avoided_packs']==1740 and row['reconstructions']==0
        assert row['scratch_reduction']==29192355840 and row['copy_increase']==0
        assert len(row['clips'])==20 and all(c['outputs_exact'] for c in row['clips'])
        assert all(c['avoided_packs']==87 and c['reconstructions']==0 and c['copy_increase']==0 for c in row['clips'])
    report=dict(passed=True,diagnostic_only=True,performance_measured=False,release_admitted=False,
        closure=pin(BASE/'closed.json'),analysis=pin(BASE/'analysis.json'),products=analysis['products'],
        model_closure=analysis['model_closure'],census_closure=analysis['census_closure'],comparisons=analysis['comparisons'],
        resources=analysis['resources'],failed_release_controls=analysis['failed_release_controls'],publisher=pin(Path(__file__)))
    prose='''# Parakeet: zero reconstruction on the actual corpus

**The candidate performs zero feed-forward weight reconstructions on all 20
clips in both instruction modes.** It retains the same 1,740 avoided transient
16 MiB packs per corpus: 29,192,355,840 fewer cumulative scratch bytes than
selected M73, with zero additional copy bytes. Every per-clip prediction matches.

The seven affected sequence lengths are 167, 89, 157, 83, 61, 151 and 169.
The remaining row now reads prepared weights directly. The existing two-/three-row
routes, nine already prepared feed-forward weights and all 37 packing records
remain intact. All 96 feed-forward calls per request report zero CopyY events,
including the previously reconstructing paths.

Across 80 encoder requests and 7,680 feed-forward calls, frontend and encoder
outputs are hash-identical between selected M73 and candidate M78. Audio and
features remain immutable; held outputs remain independent. Actual dimensions,
all 2,856 ordered profile records, product identities, instruction availability,
affinity and resource bounds pass. Core49901366/Data01e9e784 are unchanged from
full-model qualification. Only the diagnostic consumer was built, with one
prospective assertion change requiring zero copies. Profiler durations are
discarded; no application latency is scored.

The four fresh processes run selected/candidate in normal and AVX512-disabled
modes. The existing bounded memory-preflight wait retains its observations and
does not lower any threshold. Every completed worker and raw result is retained.

These are cumulative traffic counts, not peak memory, hardware bandwidth or
predicted application savings. Next, run the matched six-process comparison
against fresh ORT, with the original 3% gain and repeatability gates. The prior
e5 release failures remain; BENCHMARK.md stays on the qualified product.

[Every per-clip count, product identity and resource result](counters-20260925.json),
[full-model correctness](models-20260925.md),
[counter protocol](../packed-final-row-counters/README.md).
'''
    write(paths[0],report)
    with paths[1].open('x',encoding='utf8') as stream:stream.write(prose)
    print(json.dumps(dict(passed=True,closure=pin(BASE/'closed.json'),avoided_packs_per_corpus=1740,
        reconstructions_per_corpus=0,reports={p.name:pin(p) for p in paths})))


if __name__=='__main__':main()
