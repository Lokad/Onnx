"""Close production integration and snapshot editable source after validation."""
import importlib.util
import json
from pathlib import Path
import shutil
from run import ROOT, BASE, CANDIDATE, RUNTIME, pin, read, save, verify, relative, suite

spec = importlib.util.spec_from_file_location('integration_resource_audit', ROOT / 'tests/pyannote/convolution-portable-qualification/common.py')
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def main():
    assert not (BASE / 'closed.json').exists()
    proof = read(BASE / 'verified.json')
    assert proof['passed']
    verify(proof['files'])
    jobs = {name + '-' + stage: (8, 900, False) for name in ['cli', 'backend', 'tensors', 'bridge'] for stage in ['restore', 'build']}
    jobs.update({'instructions': (8, 900, False), 'backend-tests': (10, 900, False), 'tensors-tests': (10, 900, False),
        'package': (8, 900, False), 'consumer-restore': (8, 900, False), 'consumer-build': (8, 900, False), 'consumer': (10, 900, False)})
    resources = common.resources(BASE, 'processes.json', jobs)
    assert proof['suites'] == [suite('backend', 3290, 93), suite('tensors', 342, 0)]
    inventory = read(BASE / 'instructions.json')
    for row in inventory['observations']:
        assert row['public_surface_equal'] and not row['added'] and not row['removed'] and not row['differences']
        assert row['before_sha256'] == pin(RUNTIME / row['assembly'])['sha256']
        assert row['after_sha256'] == pin(BASE / 'runtime' / row['assembly'])['sha256']
    files = dict(proof['files'])
    snapshots = {}
    for name in read(BASE / 'admission.json')['names']:
        path = ROOT / name
        target = BASE / 'source-snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        assert pin(target) == files.pop(relative(path))
        files[relative(target)] = pin(target)
        snapshots[name] = dict(path=relative(target), **pin(target))
    save(BASE / 'source-snapshots.json', snapshots)
    files[relative(Path(__file__).resolve())] = pin(Path(__file__).resolve())
    analysis = dict(passed=True, core=proof['core'], data=proof['data'], package=proof['package'], suites=proof['suites'],
        source_snapshots=pin(BASE / 'source-snapshots.json'), instructions=proof['instructions'],
        no_new_performance_measurement=True, **resources)
    common.close(BASE, analysis, files, resources['identities'])


if __name__ == '__main__':
    main()
