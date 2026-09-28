"""Publish the closed shared/e5 correctness evidence without executing requests."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'transpose-axis-shared-amd'))
from prepare import BASE, ROOT
from protocol import pin, read

assert pin(BASE/'closed.json')['sha256'] == 'fa74521cbe415ead594903a2b76649bcd4ec2edc20d075f616ae67948e920f1b'
proof = read(BASE/'closed.json'); assert proof['passed']
for name, wanted in proof['files'].items(): assert pin(BASE/name) == wanted, name
analysis = read(BASE/'analysis.json')
assert analysis['passed'] and analysis['reference_provenance_verified'] and analysis['no_performance_measurement']
summary = {}
for name, result in analysis['results'].items():
    assert result['passed'] and result['no_performance_measurement']
    if name.startswith('candidate-'): assert all(row['exact_selected'] for row in result['rows'])
    summary[name] = dict(arrays=result['arrays'], values=result['values'],
        maximum=max(row['maximum'] for row in result['rows']),
        exact_current_arrays=sum(row['exact_selected'] is True for row in result['rows']))
assert sum(row['arrays'] for row in summary.values()) == 332
assert sum(row['values'] for row in summary.values()) == 10001628
report = dict(passed=True, performance_measured=False, shared_closure=pin(BASE/'closed.json'),
    full_analysis=pin(BASE/'analysis.json'), identities=analysis['identities'], consumer=analysis['consumer'],
    results=summary, resources=analysis['resources'], source=pin(Path(__file__)))
target = ROOT/'tests/parakeet/transpose-axis-results/shared-20260928.json'
with target.open('x', encoding='utf8') as stream:
    json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
print(json.dumps(dict(passed=True, published=pin(target), results=summary)))
