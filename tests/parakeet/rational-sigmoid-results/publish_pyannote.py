"""Publish the complete Pyannote correctness result with consumer reuse explicit."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-rational-sigmoid-pyannote-amd-20260927'


def read(path): return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def main():
    proof, value = read(BASE/'closed.json'), read(BASE/'analysis.json')
    assert proof['passed'] and value['passed'] and proof['analysis'] == pin(BASE/'analysis.json')
    for name, wanted in proof['files'].items(): assert pin(BASE/name) == wanted, name
    assert value['identities']['selected']['Lokad.Onnx.dll']['sha256'] == '8bb22038d0b4c09b56b2cdae06c49c165b8e646bc73ca28ad400f4ace0bfc659'
    assert value['identities']['candidate']['Lokad.Onnx.dll']['sha256'] == '946ddfb66492c48a0fc6078ecbe1957ac494ff5d9ff0be42259d70e66e3b1f24'
    assert value['reference_provenance_verified'] and value['no_performance_measurement']
    assert not value['consumer_rebuilt'] and value['consumer_reuse']['passed']
    assert not value['consumer_reuse']['product_rebuilt']
    guards = value['identity_guards']
    assert guards['passed'] and guards['probes'] == 4 and guards['rejection_before_output']
    assert guards['consumer']['sha256'] == 'd78c45b9059c0f22e55c5112ad12fe54b34d1e0567096f52d23b67d3b141a848'
    table = []
    for role in ['selected', 'candidate']:
        result = value['results'][role]
        assert result['passed'] and (result['arrays'], result['values'], result['public_calls']) == (18, 2917107, 16)
        native = [c for c in result['comparisons'] if c['reference'] == 'native']
        assert len(native) == 18 and all(c['failed_values'] == 0 for c in result['comparisons'])
        table.append(f"| {role} | 18 | 2,917,107 | 16 | {max(c['maximum'] for c in native):.10g} |")
    candidate = value['results']['candidate']
    assert candidate['complete_public_semantics_exact']
    production = [c for c in candidate['comparisons'] if c['reference'] == 'production']
    assert len(production) == 18
    centroids = candidate['public_centroid_comparisons']; assert len(centroids) == 16
    assert all(c['failed_values'] == 0 for r in centroids for c in r['speakers'])
    exact = sum(c['bit_identical'] for c in production)
    cross_max = max(c['maximum'] for c in production)
    centroid_max = max((c['maximum'] for r in centroids for c in r['speakers']), default=0)
    resources = value['resources']; assert len(resources) == 3
    target = OUT/'pyannote-20260927'
    assert not target.with_suffix('.md').exists() and not target.with_suffix('.json').exists()
    markdown = f'''# Rational sigmoid: complete Pyannote correctness

**Both products pass the original native and ownership checks.** Public speaker
assignments, intervals, status, duration, window count and embedding state remain
exact across current root and candidate.

| Product | Arrays | Values | Public calls | Maximum scaled native array error |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(table)}

Candidate/current float comparisons were prospectively bounded at the original
scaled error 1e-4, with all differences retained. **{exact}/18 arrays are byte-identical**;
maximum scaled array difference is {cross_max:.10g}. Maximum centroid difference
is {centroid_max:.10g}. Complete public results, including centroids, are
{'exactly equal' if candidate['complete_public_results_exact'] else 'numerically bounded with exact discrete semantics'}.
Inputs and held outputs remain unchanged across later calls.

The previously qualified GraphQualification `d78c45b9` is reused byte for byte
with current Core `8bb22038` / Data `d02dbf55` and candidate `946ddfb6` / `dbe95936`.
Its existing expected-assembly arguments are checked in four fresh wrong-Core/
wrong-Data probes. All reject at the corresponding identity guard before output
creation, and all probe identities are independently confirmed terminal.
The prior consumer inventory remains prior evidence; no consumer or product was
rebuilt and no new compiled-consumer comparison is claimed.

All three jobs and their supervisor are terminal with code zero. All
{sum(r['samples'] for r in resources):,} resource observations pass; peak owned RSS
is {max(r['peak_rss'] for r in resources):,} bytes. This is correctness qualification,
not a fresh Pyannote performance measurement.

The [Parakeet application gain](application-20260927.md) and
[shared/e5 correctness](shared-20260927.md) remain qualified for this candidate.
The [operator screen](screen-20260927.md) remains rejected, with its
[double-latency limitation](fallback-diagnosis-20260927.md) unresolved.
Graph and complete Pyannote application regressions, then actual-root/package
qualification, remain necessary before source or BENCHMARK.md promotion.

[All identities, guard results, numerical comparisons and resources](pyannote-20260927.json).
[Prospective protocol](../rational-sigmoid-pyannote-amd/README.md).
Closure: `{pin(BASE/'closed.json')['sha256']}`.
Raw evidence: `{BASE.relative_to(ROOT).as_posix()}`.
'''
    with target.with_suffix('.json').open('x', encoding='utf8') as stream:
        json.dump(dict(closure=pin(BASE/'closed.json'), **value, release_admitted=False), stream, indent=2, allow_nan=False)
        stream.write('\n')
    with target.with_suffix('.md').open('x', encoding='utf8') as stream: stream.write(markdown)
    print(json.dumps(dict(passed=True, exact_candidate_arrays=exact,
        complete_public_results_exact=candidate['complete_public_results_exact'], report=pin(target.with_suffix('.md')))))


if __name__ == '__main__': main()
