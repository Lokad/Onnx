"""Close every build/test sample and independently confirm the one-method change."""
from common import *


def main():
    assert not (BASE / 'focused-closed.json').exists()
    prepared, initial = read(BASE / 'prepared.json'), read(BASE / 'source-prepared.json')
    assert prepared['passed'] and initial['passed']
    verify(prepared['files'])
    verify(initial['files'])
    changed = []
    for path in (BASE / 'source').rglob('*'):
        name = path.relative_to(BASE / 'source')
        if path.is_file() and not {'obj', 'bin'}.intersection(name.parts):
            if pin(path) != pin(SOURCE / 'source' / name):
                changed.append(name.as_posix())
    assert changed == prepared['changed'] == initial['changed'] == ['src/Lokad.Onnx/TensorOps.ConvPool.cs']
    instructions = read(BASE / 'instructions.json')
    assert instructions['passed']
    for row in instructions['observations']:
        assert row['public_surface_equal'] and not row['removed'] and not row['added']
        if row['assembly'] == 'Lokad.Onnx.dll':
            assert len(row['differences']) == 1 and row['differences'][0].startswith('Lokad.Onnx.Tensor`1[T]::RunTiledBatchFloat::')
        else:
            assert row['assembly'] == 'Lokad.Onnx.Data.dll' and not row['differences']
        assert row['before_sha256'] == pin(PRIOR / 'runtime' / row['assembly'])['sha256']
        assert row['after_sha256'] == pin(BASE / 'runtime' / row['assembly'])['sha256']
    suites = [read_suite(name, passed, skipped) for name, _, _, passed, skipped in SUITES]
    assert suites == prepared['suites'] == read(BASE / 'suites.json')
    expected = [name + '-' + stage for name in ['cli', 'backend', 'tensors', 'bridge'] for stage in ['restore', 'build']]
    expected += ['instructions'] + [s[0] for s in SUITES]
    resources = audit_resources(BASE / 'processes.json', expected)
    analysis = dict(passed=True, suites=suites, changed=changed, core=prepared['core'], data=prepared['data'],
        all_other_methods_and_public_declarations_unchanged=True, **resources, scope=prepared['scope'])
    save(BASE / 'focused-analysis.json', analysis)
    files = dict(prepared['files'])
    for path in BASE.rglob('*'):
        if path.is_file() and not {'obj', 'packages'}.intersection(path.relative_to(BASE).parts):
            files[rel(path)] = pin(path)
    save(BASE / 'focused-closed.json', dict(passed=True, files=files, identities=resources['identities'], analysis=pin(BASE / 'focused-analysis.json')))
    print(json.dumps(dict(closed=pin(BASE / 'focused-closed.json'), core=prepared['core'], data=prepared['data'],
        resource_samples=resources['resource_samples'], peak_rss=resources['peak_rss'])))


if __name__ == '__main__':
    main()
