"""Close root integration after the explicit historical-fixture policy correction."""
import importlib.util
from pathlib import Path
import shutil
from resume import ROOT, BASE, PRIOR, pin, read, save, verify, rel, predecessor, suite

spec = importlib.util.spec_from_file_location('completion_resource_audit', ROOT / 'tests/pyannote/convolution-portable-qualification/common.py')
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def main():
    assert not (BASE / 'closed.json').exists()
    failed = predecessor()
    proof = read(BASE / 'verified.json')
    assert proof['passed']
    verify(proof['files'])
    assert proof['backend'] == suite(PRIOR / 'test-results/backend.trx', 3290, 93)
    assert proof['tensors'] == suite(BASE / 'test-results/tensors.trx', 343, 0)
    jobs = {name: (minimum, 900, False) for name, minimum in [('tensor-build', 8), ('tensor-tests', 10),
        ('package', 8), ('consumer-restore', 8), ('consumer-build', 8), ('consumer', 10)]}
    resources = common.resources(BASE, 'processes.json', jobs)
    files = dict(proof['files'])
    snapshots = {}
    for name in read(BASE / 'inputs.json')['source_names']:
        source = ROOT / name; target = BASE / 'source-snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        assert pin(target) == files.pop(rel(source))
        files[rel(target)] = pin(target); snapshots[name] = dict(path=rel(target), **pin(target))
    save(BASE / 'source-snapshots.json', snapshots)
    analysis = dict(passed=True, core=proof['core'], data=proof['data'], package=proof['package'],
        backend=proof['backend'], tensors=proof['tensors'], instructions=proof['instructions'],
        source_snapshots=pin(BASE / 'source-snapshots.json'), failure=pin(PRIOR / 'failure-closed.json'),
        prior_samples=failed['samples'], total_samples=failed['samples'] + resources['samples'],
        total_identities=len(failed['identities']) + len(resources['identities']),
        policy_correction=proof['policy_correction'], no_new_performance_measurement=True, **resources)
    common.close(BASE, analysis, files, resources['identities'])


if __name__ == '__main__':
    main()
