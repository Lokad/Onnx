"""Publish the completed fixed candidate's complete shared/e5 qualification once."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-rational-sigmoid-shared-amd-20260927'


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
    assert value['consumer']['sha256'] == 'a50d3e965cf480844559b1f5856e3de318b5c8a269742a9e6504afb9cee6c8c0'
    assert value['reference_provenance_verified'] and value['no_performance_measurement']
    candidates = []
    table = []
    for role in ['selected', 'candidate']:
        for mode in ['shared', 'e5']:
            rows = value['results'][role+'-'+mode]
            assert rows['passed'] and rows['no_performance_measurement']
            assert rows['arrays'] == len(rows['rows']) == (106 if mode == 'shared' else 60)
            assert all(0 <= r['maximum'] <= 1e-4 for r in rows['rows'])
            table.append(f"| {role} / {mode} | {rows['arrays']} | {rows['values']:,} | {max(r['maximum'] for r in rows['rows']):.10g} |")
            if role == 'candidate': candidates.extend(rows['rows'])
        assert sum(value['results'][role+'-'+m]['values'] for m in ['shared', 'e5']) == 5000814
    assert len(candidates) == 166
    for row in candidates:
        c = row['selected_comparison']
        assert c['values'] == row['values'] and 0 <= c['maximum_scaled_error'] <= 1e-4
        assert c['bit_identical'] == row['exact_selected']
    exact = sum(row['exact_selected'] for row in candidates)
    maximum = max(row['selected_comparison']['maximum_scaled_error'] for row in candidates)
    resources = value['resources']; assert len(resources) == 4
    target = OUT/'shared-20260927'
    assert not target.with_suffix('.md').exists() and not target.with_suffix('.json').exists()
    markdown = f'''# Rational sigmoid: shared-model and e5 correctness

**Both products pass all 166 arrays / 5,000,814 values.** The original shared
models and all five e5 shapes/padding cases pass native numerical, input-ownership,
held-output, execution-context reuse and memory-policy checks.

| Product / scope | Arrays | Values | Maximum scaled ORT error |
| --- | ---: | ---: | ---: |
{chr(10).join(table)}

The unchanged native bound is `abs(actual-reference)/max(1,abs(reference)) <= 1e-4`.
Candidate/current comparisons were prospectively given the same numerical bound
for the new arithmetic, with every difference recorded. **{exact}/166 arrays are
byte-identical**; the maximum scaled candidate/current difference is {maximum:.10g}.

The original Replay `a50d3e96` runs unchanged with current Core `8bb22038` and
candidate Core `946ddfb6`. No Data assembly is loaded. No product or consumer was
rebuilt, and no model or package was downloaded. All four jobs and their supervisor
are terminal with code zero. All {sum(r['samples'] for r in resources):,} resource
observations pass; peak owned RSS is {max(r['peak_rss'] for r in resources):,} bytes.

This is correctness qualification, with no performance score. The
[Parakeet application gain](application-20260927.md) remains 3.249%, or 1.244628
times ORT. The [operator screen](screen-20260927.md) remains rejected, including
its fallback regressions; the [double-latency limitation](fallback-diagnosis-20260927.md)
remains unresolved. Pyannote, graph and actual-root/package qualification are
still required before source or BENCHMARK.md promotion.

[Every numerical comparison, product identity and resource result](shared-20260927.json).
[Protocol and prospective bounds](../rational-sigmoid-shared-amd/README.md).
Closure: `{pin(BASE/'closed.json')['sha256']}`.
Raw evidence: `{BASE.relative_to(ROOT).as_posix()}`.
'''
    with target.with_suffix('.json').open('x', encoding='utf8') as stream:
        json.dump(dict(closure=pin(BASE/'closed.json'), **value, release_admitted=False), stream, indent=2, allow_nan=False)
        stream.write('\n')
    with target.with_suffix('.md').open('x', encoding='utf8') as stream: stream.write(markdown)
    print(json.dumps(dict(passed=True, arrays_per_product=166, exact_candidate_arrays=exact,
                         maximum_candidate_difference=maximum, report=pin(target.with_suffix('.md')))))


if __name__ == '__main__': main()
