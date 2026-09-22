"""Audit root integration and freeze source snapshots after successful checks."""
import importlib.util
import json
from pathlib import Path
import shutil
from run import ROOT, TOOLS, BASE, AMD, MODEL, NAMES, pin, read, save, verify, rel, suite

spec = importlib.util.spec_from_file_location('direct_root_resource_auditor', ROOT / 'tests/parakeet/portable-models/common.py')
common = importlib.util.module_from_spec(spec); spec.loader.exec_module(common)


def main():
    assert not (BASE / 'closed.json').exists()
    proof = read(BASE / 'verified.json'); assert proof['passed']; verify(proof['files'])
    jobs = {n + '-' + p: (8, 8, 900, False) for n in ['cli', 'backend', 'tensors', 'bridge'] for p in ['restore', 'build']}
    jobs.update({n: (minimum, 8, 900, False) for n, minimum in [('instructions', 8), ('backend-tests', 10), ('tensors-tests', 10),
                 ('package', 8), ('consumer-restore', 8), ('consumer-build', 8), ('consumer', 10)]})
    resources = common.resources(BASE, 'processes.json', jobs)
    assert proof['suites'] == [suite('backend', 3313, 93), suite('tensors', 343, 0)]
    inventory = read(BASE / 'instructions.json'); assert inventory['inventory_complete']
    assert [(r['assembly'], r['methods']) for r in inventory['observations']] == [('Lokad.Onnx.dll', 3114), ('Lokad.Onnx.Data.dll', 697)]
    for row in inventory['observations']:
        assert row['public_surface_equal'] and not row['added'] and not row['removed'] and not row['differences']
        assert row['before_sha256'] == pin(MODEL / 'runtime' / row['assembly'])['sha256']
        assert row['after_sha256'] == pin(BASE / 'runtime' / row['assembly'])['sha256']
    value = read(BASE / 'consumer.json')
    assert value['passed'] and value['packaged_tiled_convolution_values'] == 33216
    assert value['packaged_narrow_values'] == 10240 and value['narrow_scratch_bytes'] == 327680
    assert value['core'] == proof['core']['sha256'] and value['processor_count'] == 1
    assert value['model_imported'] and value['input_and_held_outputs_unchanged']
    assert value['pid'] == read(BASE / 'processes.json')['runs'][-1]['worker']['pid']
    files = dict(proof['files']); snapshots = {}
    for name in NAMES:
        source = ROOT / name; target = BASE / 'source-snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        assert pin(target) == files.pop(rel(source))
        files[rel(target)] = pin(target); snapshots[name] = dict(path=rel(target), **pin(target))
    save(BASE / 'source-snapshots.json', snapshots)
    result = dict(passed=True, core=proof['core'], data=proof['data'], package=proof['package'], suites=proof['suites'],
                  dependencies=proof['dependencies'], instructions=proof['instructions'], source_snapshots=pin(BASE / 'source-snapshots.json'),
                  amd_closure=pin(AMD / 'closed.json'), no_new_performance_measurement=True,
                  resource_samples=sum(r['samples'] for r in resources['resources']), **resources)
    save(BASE / 'analysis.json', result)
    for p in BASE.rglob('*'):
        if p.is_file() and not {'obj', 'packages', 'consumer-cache'}.intersection(p.relative_to(BASE).parts): files[rel(p)] = pin(p)
    save(BASE / 'closed.json', dict(passed=True, files=files, identities=resources['identities'], analysis=pin(BASE / 'analysis.json')))
    print(json.dumps(dict(closed=pin(BASE / 'closed.json'), core=proof['core'], data=proof['data'], package=proof['package'], resources=result['resource_samples'])))


if __name__ == '__main__': main()
