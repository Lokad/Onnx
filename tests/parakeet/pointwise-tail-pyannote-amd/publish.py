"""Publish closed Pyannote correctness without inferring speed from its profiles."""
import json
from pathlib import Path
from prepare import BASE, ROOT
from protocol import pin, read

assert pin(BASE/'closed.json')['sha256'] == 'f284cc72de08b0113daa8a5482e2a2dbea2bd6a89ea00e128a569036394eebb8'
proof = read(BASE/'closed.json'); assert proof['passed']
for name, wanted in proof['files'].items(): assert pin(BASE/name) == wanted, name
analysis = read(BASE/'analysis.json')
assert analysis['passed'] and analysis['no_performance_measurement'] and analysis['reference_provenance_verified']
summary = {}
for role, result in analysis['results'].items():
    assert result['passed'] and result['no_performance_measurement']
    native = [row for row in result['comparisons'] if row['reference'] == 'native']
    exact = [row for row in result['comparisons'] if row['reference'] == 'production']
    assert len(native) == 18 and not any(row['failed_values'] for row in native)
    assert len(exact) == (18 if role == 'candidate' else 0)
    assert all(row['bit_identical'] and row['maximum'] == 0 for row in exact)
    if role == 'candidate': assert result['complete_public_results_exact'] and result['complete_public_semantics_exact']
    summary[role] = dict(arrays=result['arrays'], values=result['values'], public_calls=result['public_calls'],
        native_maximum=max(row['maximum'] for row in native), exact_current_arrays=len(exact),
        complete_public_results_exact=result['complete_public_results_exact'])
assert sum(row['arrays'] for row in summary.values()) == 36
assert sum(row['values'] for row in summary.values()) == 5834214
assert sum(row['public_calls'] for row in summary.values()) == 32
report = dict(passed=True, performance_measured=False, pyannote_closure=pin(BASE/'closed.json'),
    full_analysis=pin(BASE/'analysis.json'), identities=analysis['identities'], consumers=analysis['consumers'],
    identity_guards=analysis['identity_guards'], results=summary, resources=analysis['resources'], source=pin(Path(__file__)))
target = ROOT/'tests/parakeet/decoder-lstm-layout-profile-results/pointwise-tail-pyannote-observations-20260927.json'
with target.open('x', encoding='utf8') as stream:
    json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
print(json.dumps(dict(passed=True, published=pin(target), results=summary)))
