"""Review the completed bounded build before permitting any diagnostic inference."""
import base64
from collections import Counter
import json
import re

from build_checks import inventory
from run import BASE, ROOT, RELEASE, PRELUDE, pin, read, write, ssh, prerequisites

JOBS = ['sdk-version', 'core-restore', 'core-build', 'data-restore', 'data-build', 'inventory']


def warnings(folder, job):
    result = []
    for suffix in ['stdout', 'stderr']:
        for line in (folder / 'logs' / (job + '.' + suffix)).read_text(encoding='utf8').splitlines():
            if ': warning ' not in line:
                continue
            match = re.search(r'(Zzz\.WideProjectionEntry\.cs\(\d+,\d+\): warning CS8604:.*?) \[', line)
            assert match is not None, ('Unexpected warning', job, line)
            result.append(match.group(1))
    return Counter(result)


def main():
    _, _, _, _, isolated = prerequisites()
    assert not (BASE / 'build-review.json').exists(), 'Preserve the original build verdict'
    folder = BASE / 'build-collected'
    spec = read(BASE / 'bundle/spec.json')
    receipt = read(folder / 'build-collection.json')
    transfer = read(BASE / 'build-transfer.json')
    assert transfer['passed'] and transfer['archive'] == pin(BASE / 'build-results.tar.gz')
    assert transfer['collection'] == pin(folder / 'build-collection.json')
    assert receipt['terminal'] and receipt['code'] == 0
    for name, wanted in receipt['files'].items():
        assert pin(folder / name) == wanted, name
    assert pin(folder / 'spec.json') == pin(BASE / 'bundle/spec.json')
    for name, wanted in spec['files'].items():
        assert pin(folder / name) == wanted, name
    state = read(folder / 'build-state.json')
    assert state['complete'] and state['code'] == 0 and receipt['state'] == pin(folder / 'build-state.json')
    assert state['supervisor'] == read(BASE / 'build-deployment.json')
    assert [r['name'] for r in state['runs']] == JOBS
    resources = []
    limits = spec['build_limits']
    for row in state['runs']:
        assert row['complete'] and row['code'] == 0 and row['seconds'] < limits['seconds']
        assert row['preflight']['available'] >= limits['available_before'] and row['preflight']['tmpfs'] >= limits['tmpfs_before']
        samples = [json.loads(line) for line in (folder / 'logs' / (row['name'] + '.resources.jsonl')).read_text().splitlines()]
        assert len(samples) == row['samples'] > 0
        for sample in samples:
            assert sample['seconds'] < limits['seconds'] and sample['rss'] < limits['rss']
            assert sample['available'] >= spec['minimum_free'] and sample['tmpfs'] >= spec['minimum_free']
            assert sample['output'] < spec['output_limit']
            assert sample['rss'] == sum(m['rss'] for m in sample['members'])
            for member in sample['members']:
                assert row['members'][str(member['pid'])] == member['birth']
                assert member['affinity'] == [2] and member['threads'] and all(t == [2] for t in member['threads'])
        gaps = [samples[0]['seconds']] + [b['seconds'] - a['seconds'] for a, b in zip(samples, samples[1:])] + [row['seconds'] - samples[-1]['seconds']]
        assert all(0 <= gap < 10 for gap in gaps)
        resources.append(dict(name=row['name'], samples=len(samples), peak_rss=max(s['rss'] for s in samples), seconds=row['seconds']))
    assert (folder / 'logs/sdk-version.stdout').read_text().strip() == '10.0.204'
    built = read(folder / 'built.json')
    assert built['passed'] and built['consumer'] == spec['consumer'] and built['original_core'] == spec['product']['Lokad.Onnx.dll']
    for name, wanted in built['runtime_files'].items():
        assert pin(folder / name) == wanted, name
    for name, wanted in spec['original_runtime_files'].items():
        assert pin(folder / 'runtime-original' / name) == wanted, name
    assert pin(folder / 'runtime-control/Lokad.Onnx.dll') == spec['product']['Lokad.Onnx.dll']
    for role in ['control', 'observed']:
        assert pin(folder / ('runtime-' + role) / 'Lokad.Onnx.Data.dll') == built['data']
        assert pin(folder / ('runtime-' + role) / 'SampledAudio.dll') == spec['consumer']
    certificate = read(ROOT / 'tests/parakeet/slice-dense-conversion-results/observer-20260925.json')
    original_observer = ROOT / 'artifacts/parakeet-managed-phase-amd-20260924/build-collected/inventory/instructions.json'
    assert pin(original_observer) == certificate['original_inventory']
    assert spec['isolated_evidence'] == isolated['evidence'] and not spec['release_admitted'] and spec['diagnostic_only']
    assert read(folder / 'evidence/isolated-baseline.json') == {k:v for k,v in isolated.items() if k != 'inventory'}
    il = inventory(read(folder / 'inventory/instructions.json'), isolated['inventory'], read(original_observer), spec['product'], built)
    baseline_warnings = warnings(RELEASE / 'collected', 'cli-build')
    assert sum(baseline_warnings.values()) == 4 and len(baseline_warnings) == 2
    assert warnings(folder, 'core-build') == baseline_warnings
    assert not warnings(folder, 'data-build')
    for job in ['core-restore', 'data-restore']:
        assert not warnings(folder, job)
    result = dict(passed=True, arithmetic_equivalent=True, source_changes_exact=True,
        built=pin(folder / 'built.json'), inventory=pin(folder / 'inventory/instructions.json'),
        isolated_evidence=isolated['evidence'], original_observer_inventory=pin(original_observer),
        spec=pin(BASE / 'bundle/spec.json'), collection=pin(folder / 'build-collection.json'),
        selected_release=spec['selected_release_closure'], release_admitted=False, diagnostic_only=True,
        methods=il, resources=resources,
        warnings=dict(baseline_warnings), consumer_rebuilt=False, reviewer=pin(__file__))
    write(BASE / 'build-review.json', result)
    encoded = base64.b64encode((BASE / 'build-review.json').read_bytes()).decode()
    remote = ssh(PRELUDE + f'''
import base64
from remote import verify,pin,read,live
verify();state=read(base/'build-state.json')
assert state['complete'] and state['code']==0 and not live(state['supervisor'])
assert all(not live(dict(pid=int(p),birth=b)) for r in state['runs'] for p,b in r['members'].items())
assert pin(base/'built.json')=={result['built']!r} and pin(base/'inventory/instructions.json')=={result['inventory']!r}
with (base/'build-review.json').open('xb') as stream:stream.write(base64.b64decode({encoded!r}))
print(json.dumps(dict(passed=True,review=pin(base/'build-review.json'))))
''')
    assert remote['review'] == pin(BASE / 'build-review.json')
    write(BASE / 'build-review-transferred.json', remote)
    print(json.dumps(dict(passed=True, review=remote['review'], core=built['core'], data=built['data'],
                         consumer=built['consumer'], original_warnings=4, arithmetic_equivalent=True)))


if __name__ == '__main__':
    main()
