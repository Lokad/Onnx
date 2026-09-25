"""Review both retained failures and complete scoped inventories before capture."""
import base64
from collections import Counter
import json
from pathlib import Path
import re
from run import BASE, ROOT, FIRST, OBSERVER, original, pin, read, write, ssh, PRELUDE, prepared
from build_checks import inventory


def warnings(folder, job):
    values = []
    for suffix in ['stdout', 'stderr']:
        for line in (folder / 'logs' / (job + '.' + suffix)).read_text(encoding='utf8').splitlines():
            if ': warning ' not in line: continue
            match = re.search(r'(Zzz\.WideProjectionEntry\.cs\(\d+,\d+\): warning CS8604:.*?) \[', line)
            assert match is not None, ('Unexpected warning', job, line)
            values.append(match.group(1))
    return Counter(values)


def resources(folder, state, spec):
    result = []
    limits = spec['build_limits']
    for row in state['runs']:
        assert row['complete'] and row['seconds'] < limits['seconds']
        assert row['preflight']['available'] >= limits['available_before'] and row['preflight']['tmpfs'] >= limits['tmpfs_before']
        samples = [json.loads(s) for s in (folder / 'logs' / (row['name'] + '.resources.jsonl')).read_text().splitlines()]
        assert len(samples) == row['samples'] > 0
        for sample in samples:
            assert sample['seconds'] < limits['seconds'] and sample['rss'] < limits['rss']
            assert sample['available'] >= spec['minimum_free'] and sample['tmpfs'] >= spec['minimum_free']
            assert sample['output'] < spec['output_limit']
            assert sample['rss'] == sum(m['rss'] for m in sample['members'])
            for member in sample['members']:
                assert row['members'][str(member['pid'])] == member['birth']
                assert member['affinity'] == [2] and member['threads'] and all(t == [2] for t in member['threads'])
        gaps = [samples[0]['seconds']] + [b['seconds']-a['seconds'] for a,b in zip(samples,samples[1:])] + [row['seconds']-samples[-1]['seconds']]
        assert all(0 <= gap < 10 for gap in gaps)
        result.append(dict(name=row['name'], code=row['code'], samples=len(samples), peak_rss=max(s['rss'] for s in samples), seconds=row['seconds']))
    return result


def main():
    prepared()
    _, _, _, _, isolated = original.prerequisites()
    assert not (BASE / 'review.json').exists() and not (original.BASE / 'build-review.json').exists()
    old = original.BASE / 'build-collected'; previous = FIRST / 'collected'; folder = BASE / 'collected'
    spec = read(original.BASE / 'bundle/spec.json')
    repair = read(BASE / 'bundle/observer-finalize-spec.json')
    for root, receipt_name, transfer_name in [(old, 'build-collection.json', 'build-transfer.json'),
                                             (previous, 'observer-recovery-collection.json', 'transfer.json'),
                                             (folder, 'observer-finalize-collection.json', 'transfer.json')]:
        receipt = read(root / receipt_name)
        transfer = read(root.parent / transfer_name)
        archive = root.parent / ('build-results.tar.gz' if root == old else 'results.tar.gz')
        assert transfer['passed'] and transfer['collection'] == pin(root / receipt_name) and transfer['archive'] == pin(archive)
        for name, wanted in receipt['files'].items(): assert pin(root / name) == wanted, name
        assert receipt['terminal']
        assert receipt['code'] == (0 if root == folder else 1)
    assert pin(folder / 'spec.json') == repair['original_spec'] == pin(original.BASE / 'bundle/spec.json')
    assert pin(folder / 'observer-finalize-spec.json') == pin(BASE / 'bundle/observer-finalize-spec.json')
    assert pin(folder / 'observer-recovery-spec.json') == repair['recovery_spec'] == pin(FIRST / 'bundle/observer-recovery-spec.json')
    for name, wanted in spec['files'].items(): assert pin(folder / name) == wanted, name
    for name, wanted in repair['files'].items(): assert pin(folder / name) == wanted, name
    assert spec['isolated_evidence'] == isolated['evidence'] and not spec['release_admitted']
    before, after = (old / 'data-source/PhaseProbe.cs').read_bytes(), (folder / 'observer-finalize-source.cs').read_bytes()
    intermediate = (previous / 'observer-recovery-source.cs').read_bytes()
    needle = b'                        Require(wall.All(n => start <= n.StartTicks'
    replacement = b'                        long graphStart = start;\r\n                        Require(wall.All(n => graphStart <= n.StartTicks'
    assert before.count(needle) == intermediate.count(replacement) == 1 and intermediate.replace(replacement, needle) == before
    assert intermediate.count(b'Require(wall.All(n => graphStart') == 1
    assert after.count(b'Require(wall!.All(n => graphStart') == 1
    assert after.replace(b'Require(wall!.All(n => graphStart', b'Require(wall.All(n => graphStart') == intermediate
    assert after == (folder / 'observer-finalize-source/PhaseProbe.cs').read_bytes()
    for path in (old / 'data-source').iterdir():
        if path.name != 'PhaseProbe.cs': assert pin(path) == pin(folder / 'observer-finalize-source' / path.name)
    original_state = read(old / 'build-state.json'); state = read(folder / 'observer-finalize-state.json')
    failed_state = read(previous / 'observer-recovery-state.json')
    assert failed_state['complete'] and failed_state['code'] == 1
    assert pin(previous / 'observer-recovery-state.json') == repair['recovery_state']
    assert failed_state['supervisor'] == read(FIRST / 'deployment.json')
    assert [(r['name'], r['code']) for r in failed_state['runs']] == [('observer-data-restore',0),('observer-data-build',0),('observer-inventory',-6)]
    assert 'Unexpected Data method change.' in (previous / 'logs/observer-inventory.stderr').read_text()
    assert original_state['complete'] and original_state['code'] == 1 and pin(old / 'build-state.json') == repair['original_state']
    assert original_state['supervisor'] == read(original.BASE / 'build-deployment.json')
    original_receipt = read(old / 'build-collection.json')
    assert original_receipt['state'] == pin(old / 'build-state.json')
    assert original_receipt['identities'] == [original_state['supervisor']] + [dict(pid=int(p),birth=b) for r in original_state['runs'] for p,b in r['members'].items()]
    assert [(r['name'], r['code']) for r in original_state['runs']] == [('sdk-version',0),('core-restore',0),('core-build',0),('data-restore',0),('data-build',1)]
    assert state['complete'] and state['code'] == 0 and state['supervisor'] == read(BASE / 'deployment.json')
    assert [(r['name'], r['code']) for r in state['runs']] == [('final-data-restore',0),('final-data-build',0),('final-core-inventory',0),('final-data-inventory',0)]
    receipt = read(folder / 'observer-finalize-collection.json')
    assert receipt['state'] == pin(folder / 'observer-finalize-state.json')
    assert receipt['identities'] == [state['supervisor']] + [dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    old_resources, previous_resources, new_resources = resources(old, original_state, spec), resources(previous, failed_state, spec), resources(folder, state, spec)
    assert (old / 'logs/sdk-version.stdout').read_text().strip() == '10.0.204'
    assert 'error CS1673' in (old / 'logs/data-build.stdout').read_text()
    built = read(folder / 'built.json')
    assert built['passed'] and built['core'] == repair['core'] and not repair['core_rebuilt'] and not repair['consumer_rebuilt']
    assert built['consumer'] == spec['consumer'] and built['original_core'] == spec['product']['Lokad.Onnx.dll']
    for name, wanted in built['runtime_files'].items(): assert pin(folder / name) == wanted, name
    for name, wanted in spec['original_runtime_files'].items(): assert pin(folder / 'runtime-original' / name) == wanted, name
    assert pin(folder / 'runtime-control/Lokad.Onnx.dll') == spec['product']['Lokad.Onnx.dll']
    for role in ['control', 'observed']:
        assert pin(folder / f'runtime-{role}/Lokad.Onnx.Data.dll') == built['data']
        assert pin(folder / f'runtime-{role}/SampledAudio.dll') == spec['consumer']
    certificate = read(ROOT / 'tests/parakeet/slice-dense-conversion-results/observer-20260925.json')
    old_observer = ROOT / 'artifacts/parakeet-managed-phase-amd-20260924/build-collected/inventory/instructions.json'
    assert pin(old_observer) == certificate['original_inventory']
    core_inventory = read(folder / 'inventory/core-instructions.json')
    data_inventory = read(folder / 'inventory/data-instructions.json')
    joined = read(folder / 'inventory/instructions.json')
    assert core_inventory['inventory_complete'] and data_inventory['inventory_complete'] and joined['inventory_complete']
    assert [r['assembly'] for r in core_inventory['observations']] == ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll']
    assert [r['assembly'] for r in data_inventory['observations']] == ['SampledAudio.dll', 'Lokad.Onnx.Data.dll']
    assert joined['observations'] == [core_inventory['observations'][0], data_inventory['observations'][1]]
    assert joined['components'] == {n:pin(folder / 'inventory' / n) for n in ['core-instructions.json','data-instructions.json']}
    for row, wanted in [(core_inventory['observations'][1],spec['product']['Lokad.Onnx.Data.dll']),
                        (data_inventory['observations'][0],spec['consumer'])]:
        assert row['before_sha256'] == row['after_sha256'] == wanted['sha256']
        assert row['public_surface_equal'] and not row['differences'] and not row['added'] and not row['removed']
        assert row['method_flags_before'] == row['method_flags_after']
    inspector = folder / 'observer-finalize-inspector'
    assert pin(inspector / 'Bridge.dll')['sha256'] == 'd31414683e281ae937f9d9dfac6da52764dfcdf49603e073a36766b146d24dc8'
    assert pin(inspector / 'Program.cs') == pin(OBSERVER / 'bundle/bridge-source/Program.cs')
    assert pin(inspector / 'qualified-build.json') == pin(OBSERVER / 'build-review.json')
    methods = inventory(joined, isolated['inventory'], read(old_observer), spec['product'], built)
    baseline_warnings = warnings(original.RELEASE / 'collected', 'cli-build')
    assert sum(baseline_warnings.values()) == 4 and len(baseline_warnings) == 2
    assert warnings(old, 'core-build') == baseline_warnings
    assert not warnings(folder, 'final-data-build') and not warnings(folder, 'final-data-restore')
    result = dict(passed=True, arithmetic_equivalent=True, source_changes_exact=True, diagnostic_only=True,
        release_admitted=False, original_build_failed=True, first_recovery_failed=True, core_rebuilt=False, consumer_rebuilt=False,
        original_collection=pin(old / 'build-collection.json'), collection=pin(folder / 'observer-finalize-collection.json'),
        first_recovery_collection=pin(previous / 'observer-recovery-collection.json'),
        corrected_observer=pin(folder / 'observer-finalize-source.cs'), correction=pin(folder / 'observer-finalize-source-review.json'),
        built=pin(folder / 'built.json'), inventory=pin(folder / 'inventory/instructions.json'),
        isolated_evidence=isolated['evidence'], methods=methods, original_resources=old_resources, first_recovery_resources=previous_resources, resources=new_resources,
        warnings=dict(baseline_warnings), reviewer=pin(Path(__file__)))
    write(BASE / 'review.json', result); write(original.BASE / 'build-review.json', result)
    encoded = base64.b64encode((BASE / 'review.json').read_bytes()).decode()
    transferred = ssh(PRELUDE + f'''
from remote import verify,read,pin,live
import base64
verify();state=read(base/'observer-finalize-state.json')
assert state['complete'] and state['code']==0 and not live(state['supervisor'])
assert all(not live(dict(pid=int(p),birth=b)) for r in state['runs'] for p,b in r['members'].items())
assert pin(base/'built.json')=={result['built']!r} and pin(base/'inventory/instructions.json')=={result['inventory']!r}
with (base/'build-review.json').open('xb') as stream:stream.write(base64.b64decode({encoded!r}))
print(json.dumps(dict(passed=True,review=pin(base/'build-review.json'))))
''')
    assert transferred['review'] == pin(BASE / 'review.json') == pin(original.BASE / 'build-review.json')
    write(original.BASE / 'build-review-transferred.json', transferred)
    write(BASE / 'review-transferred.json', transferred)
    print(json.dumps(dict(passed=True, review=transferred['review'], core=built['core'], data=built['data'], core_rebuilt=False)))


if __name__ == '__main__':
    main()
