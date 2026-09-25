"""Publish the single integrated proof without implying model or score admission."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-packed-final-row-build-amd-20260925'


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def main():
    closure = read(BASE/'closed.json')
    assert pin(BASE/'closed.json')['sha256'] == '6295ad30835b7a2a1694b580a8e1447a6a0cdb28828a5960fa6e0e7c3628576f'
    assert closure['passed'] and closure['analysis'] == pin(BASE/'analysis.json')
    for name, wanted in closure['files'].items():
        assert pin(BASE/name) == wanted, name
    analysis = read(BASE/'analysis.json')
    review = read(BASE/'build-review.json')
    assert analysis['passed'] and analysis['helper_matches_proof'] and review['passed']
    assert analysis['compiled_review'] == closure['compiled_review'] == pin(BASE/'build-review.json')
    assert review['product'] == analysis['product']
    assert not analysis['release_admitted'] and analysis['no_model_execution'] and analysis['no_application_score']
    assert not analysis['prior_quantitative_attribution']
    assert [(s['mode'], s['passed'], s['skipped']) for s in analysis['suites']] == [('512',41,0), ('256',41,0), ('scalar',2,0)]
    assert [(r['unchanged'], len(r['differences']), len(r['added']), len(r['removed']), len(r['compiler_renames'])) for r in review['methods']] == [(3273,3,1,1,41), (697,0,0,0,0)]
    assert review['public_surface_unchanged'] and review['existing_arithmetic_unchanged'] and review['zero_added_warnings']
    helper = review['helper']
    assert helper['proof_body'] == helper['candidate_body'] and helper['proof_flags'] == helper['candidate_flags'] == 512
    names = [OUT/('contracts-20260925'+suffix) for suffix in ['.json','.md']]
    assert not any(path.exists() for path in names)
    value = dict(closure=pin(BASE/'closed.json'), compiled_review=pin(BASE/'build-review.json'),
        source=review['source'], product=analysis['product'], proof=analysis['proof'],
        suites=analysis['suites'], methods=review['methods'], resources=analysis['resources'],
        helper_matches_proof=True, public_surface_unchanged=True, zero_added_warnings=True,
        release_admitted=False, no_model_execution=True, no_application_score=True,
        prior_quantitative_attribution=False, failed_release_controls=analysis['failed_release_controls'],
        terminal_owners=closure['terminal_owners'], publisher=pin(Path(__file__)))
    report = '''# Parakeet: integrated packed final-row contracts

**All 84 focused checks pass.** The isolated candidate computes the remaining
row directly from its existing packed weights. Both actual matrix shapes and
all seven affected sequence lengths give bit-identical results with zero
reconstruction bytes and zero weight scratch bytes in the checked graph calls.

| Actual instruction mode | Passed checks | Skipped |
|---|---:|---:|
| Normal | 41 | 0 |
| AVX512 disabled | 41 | 0 |
| All hardware intrinsics disabled | 2 | 0 |

Each suite includes loaded-product hashes, runtime, CPU affinity and instruction
availability checks. Existing ownership, logical tensor, mutation, alias,
collection and fallback contracts remain covered. The new helper is not called
when hardware intrinsics are unavailable.

The compiled audit compares the helper's full normalized IL and implementation
flags with the successful standalone proof. They match exactly. It permits
changes to the private row loop and its two callers, removal of the dense
reconstruction method, and addition of the helper. All other 3,273 Core methods
and all 697 Data methods remain unchanged. The audit explicitly accounts for
41 compiler-generated name shifts while preserving instructions, literals,
branches, stack declarations and flags. Public interfaces remain unchanged.
The build retains two established nullable warnings and introduces none.

This implements one difference identified against the installed ORT source:
ORT's row loop continues consuming prepared weights; the previous Lokad candidate
reconstructed a full 16 MiB weight matrix to handle one row. Actual earlier
counters found 609 such reconstructions across the 20-clip corpus. This candidate
has passed focused contracts; its actual model traffic is not yet measured.
The earlier unstable copy timings remain unsuitable for predicting a saving.

Core identity is 49901366 and Data identity 01e9e784. Both were built in isolation
on the VM with SDK 10.0.204 and tested on runtime 10.0.8. Peak monitored test RSS
was 727,199,744 bytes. Both stage owners and all workers are terminal; the 581
build files and 598 contract files were collected and reviewed once.

Next, reuse the existing census consumer with these products to confirm the
87 owned weights and 37 retained maps, then complete native/public model checks,
confirm zero reconstruction on the actual corpus, and run the matched application
comparison against selected M73 and fresh ORT. The original performance gates
and unresolved e5 release controls remain. No new application speedup or release
admission is claimed; BENCHMARK.md remains on the qualified product.

[Exact identities and test names](contracts-20260925.json),
[standalone proof](proof-20260925.md),
[build and audit tools](../packed-final-row-build/README.md).

Closure: `6295ad30835b7a2a1694b580a8e1447a6a0cdb28828a5960fa6e0e7c3628576f`.
'''
    with names[0].open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    with names[1].open('x', encoding='utf8') as stream:
        stream.write(report)
    print(json.dumps(dict(passed=True, checks=sum(s['passed'] for s in analysis['suites']),
        reports={p.name:pin(p) for p in names})))


if __name__ == '__main__':
    main()
