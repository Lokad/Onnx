"""Check the exact component change, unchanged cases and terminal resources."""
import importlib.util
import json
from prepare import ROOT, TOOLS, BASE, PRIOR, MONITOR, CORE, consumer, rel, pin, read, save, verify

spec = importlib.util.spec_from_file_location('single_panel_resource_auditor', ROOT / 'tests/parakeet/portable-models/common.py')
common = importlib.util.module_from_spec(spec); spec.loader.exec_module(common)


def main():
    assert not (BASE / 'closed.json').exists()
    prepared = read(BASE / 'prepared.json'); assert prepared['passed']; verify(prepared['files'])
    assert pin(BASE / 'inputs.json') == prepared['inputs']
    inputs = read(BASE / 'inputs.json')['files']; verify(inputs)
    assert pin(BASE / 'shapes.json') == pin(PRIOR / 'payload/shapes.json')
    shapes = read(BASE / 'shapes.json'); assert len(shapes['shapes']) == 22 and len(shapes['cases']) == 3266
    assert pin(BASE / 'runtime/Lokad.Onnx.dll')['sha256'] == CORE
    for current, prior, names in [('normal', 'normal', ['DirectOutput.cs']),
                                  ('scalar', 'consumer', ['ScalarDirectOutput.cs', 'ScalarOriginal.cs'])]:
        assert (BASE / current / 'Probe.cs').read_text() == consumer((PRIOR / prior / 'Probe.cs').read_text())
        for name in names: assert pin(BASE / current / name) == pin(PRIOR / prior / name)
    state = read(BASE / 'preparation.json')
    jobs = {name + '-' + phase: (8, 4, 900, False) for name in ['normal', 'scalar', 'layout'] for phase in ['restore', 'build']}
    jobs.update({name: (8, 4, 900, True) for name in ['layout-normal', 'layout-disabled', 'validate-normal', 'validate-scalar']})
    resources = common.resources(BASE, 'preparation.json', jobs)
    layouts, validations = [], []
    for mode in ['normal', 'disabled']:
        value = read(BASE / 'output' / ('layout-' + mode + '.json'))
        assert value['passed'] and value['mode'] == mode and value['core'] == CORE
        assert value['cases'] == 1400 and value['identities'] == 660 and value['distinct_wide_cases'] == 296
        assert value['processor_count'] == 1 and value['fma'] == (mode == 'normal')
        assert value['flags'] == ([] if mode == 'normal' else ['DOTNET_EnableHWIntrinsic'])
        assert value['executable'] == pin(BASE / 'runtime/PackingLayout.dll')['sha256']
        assert value['pid'] == next(r['worker']['pid'] for r in state['runs'] if r['name'] == 'layout-' + mode)
        layouts.append(value)
    for mode, assembly in [('normal', 'DirectOutputProbe'), ('scalar', 'ScalarTailProbe')]:
        value = read(BASE / 'output' / ('validate-' + mode + '.json'))
        assert value['passed'] and value['mode'] == 'validate' and value['core'] == CORE
        assert value['executable'] == pin(BASE / 'runtime' / (assembly + '.dll'))['sha256']
        assert value['processor_count'] == 1 and value['flags'] == []
        assert value['pid'] == next(r['worker']['pid'] for r in state['runs'] if r['name'] == 'validate-' + mode)
        assert len(value['records']) == 3266
        for expected, actual in zip(shapes['cases'], value['records'], strict=True):
            assert all(expected[k] == actual[k] for k in expected), expected
        validations.append(dict(mode=mode, cases=len(value['records']), result=pin(BASE / 'output' / ('validate-' + mode + '.json'))))
    analysis = dict(passed=True, layouts=layouts, validations=validations, **resources,
                    no_performance_measurement=True, no_product_source_change=True)
    save(BASE / 'analysis.json', analysis)
    files = dict(inputs)
    files.update({rel(p): pin(p) for p in BASE.rglob('*') if p.is_file() and not {'obj', 'packages'}.intersection(p.relative_to(BASE).parts)})
    save(BASE / 'closed.json', dict(passed=True, files=files, identities=resources['identities']))
    print(json.dumps(dict(passed=True, closure=pin(BASE / 'closed.json'), layout_cases_per_mode=1400,
         component_cases_per_mode=3266, resources=sum(r['samples'] for r in resources['resources']))))


if __name__ == '__main__': main()
