"""Publish the corrected scope contracts and actual-model census; no timing claim."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'owned-packed-weight-scope-census'))
from run import BASE,CONTRACTS,PRODUCT,pin,read,references


def closed(folder,expected):
    assert pin(folder/'closed.json')['sha256']==expected
    proof=read(folder/'closed.json');assert proof['passed'] and proof['analysis']==pin(folder/'analysis.json')
    for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
    return read(folder/'analysis.json')


def main():
    assert not (HERE/'scope-20260925.json').exists() and not (HERE/'scope-20260925.md').exists()
    references()
    tests=closed(CONTRACTS,'0f8bdc94e2aaf27956af6767093acdb8b57c31655713c9af1a6302d2f7aa31ae')
    census=closed(BASE,'577a07a60d901cf57c4ba46d70b2c6fc4c02148ab67a1c28fd78d9b4fbcc3d08')
    compiled=read(PRODUCT/'build-review.json')
    assert compiled['passed'] and [len(r['differences']) for r in compiled['methods']]==[1,0]
    assert compiled['product']==tests['product']==census['product']
    assert [(s['mode'],s['passed'],s['skipped']) for s in tests['suites']]==[('512',27,0),('256',27,0),('scalar',2,0)]
    assert not census['release_admitted'] and not census['application_scored']
    report=dict(passed=True,diagnostic_only=True,release_admitted=False,performance_measured=False,
        source=compiled['source'],product=compiled['product'],compiled_review=pin(PRODUCT/'build-review.json'),
        contracts=pin(CONTRACTS/'closed.json'),census=pin(BASE/'closed.json'),
        suites=[{k:s[k] for k in ['mode','passed','skipped']} for s in tests['suites']],
        modes=[dict(mode=m['mode'],result=m['result'],memory_before=m['before']['memory'],memory_after=m['after']['memory']) for m in census['modes']],
        resources=census['resources'],failed_release_controls=census['failed_release_controls'],
        full_model_numerics_pending=True,full_public_corpus_pending=True,actual_reconstruction_census_pending=True,
        publisher=pin(Path(__file__)))
    (HERE/'scope-20260925.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    prose='''# Parakeet packed weights: corrected scope and actual model

The isolated candidate now converts exactly the intended 87 feed-forward weights.
Both normal and AVX512-disabled execution preserve all 37 existing packed cache
entries and their 256 MiB budget, all 649 initializers, original logical weight
bits, shared-context identities and immutable audio input. The longest public
clip (1188-133604-0002) returns the expected complete transcript, tokens, frames,
durations and stop reason in both modes. This is correctness evidence; no
application speedup or release admission is claimed.

The first actual census found 88 conversions: a same-shaped preprocessing
projection also matched the original predicate. The correction restricts the
private preparation to the measured feed-forward family. Compiled review proves
that only PrepareOwnedMatMulWeights differs from the initial candidate: the
other 3,276 Core methods and all 697 Data methods are unchanged. Arithmetic,
public declarations and method flags remain unchanged.

The added preprocessing regression initially failed compilation because Node is
a value type. The corrected test copies, edits and replaces that value; the
tests-only recovery reuses the already compiled product DLLs. All 27 focused
tests pass in each SIMD mode, and both hardware-disabled checks pass, without
skips. The failed compilation and original failed census remain retained.

The actual-model census uses the same previously compiled consumer. Its two
workers each finish in about 12.1 seconds; peak process RSS is 6.47 and 6.55 GB.
These diagnostic durations are not application benchmark scores. Preparation
is idempotent, the 87 replacement payloads total 1,459,617,792 bytes, and no
forced collection is used. One complete public request runs per mode.

Next are the unchanged full native and public-corpus checks, actual reconstruction
accounting, and the matched application comparison. Prior e5 repeatability
failures still prevent promotion. BENCHMARK.md continues to describe the
qualified release, not this isolated candidate.

Products: Core82c02785 / Data3f80f8cb. Compiled review5a84c53b, focused-contract
closure0f8bdc94, actual-model census577a07a6. Full identities and memory snapshots
are in [scope-20260925.json](scope-20260925.json). The corrected test source for
eventual integration is in the scope-recovery bundle; the original scope-source
archive intentionally retains its failed test source. Local artifacts are
under artifacts/parakeet-owned-packed-weight-scope-recovery-amd-20260925 and
artifacts/parakeet-owned-packed-weight-scope-census-amd-20260925.
'''
    (HERE/'scope-20260925.md').write_text(prose,encoding='utf8')
    print(json.dumps(dict(report=pin(HERE/'scope-20260925.json'),census=report['census'])))


if __name__=='__main__':main()
