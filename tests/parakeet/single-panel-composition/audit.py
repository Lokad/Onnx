"""Independently close normal-source arithmetic composition and operator proof."""
import importlib.util
from prepare import ROOT, BASE, pin, read, verify, inspect

spec = importlib.util.spec_from_file_location('composition_resource_audit', ROOT / 'tests/pyannote/convolution-portable-qualification/common.py')
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def main():
    assert not (BASE / 'closed.json').exists()
    proof = read(BASE / 'prepared.json')
    assert proof['passed'] and not proof['model_qualified'] and not proof['performance_qualified'] and not proof['production_changed']
    verify(proof['files'])
    inspect()
    jobs = {name: (8, 900, False) for name in ['cli-restore', 'cli-build', 'bridge-restore', 'bridge-build',
        'instructions', 'probe-restore', 'probe-build', 'geometry', 'hardware-off']}
    resources = common.resources(BASE, 'processes.json', jobs)
    geometry, hardware = read(BASE / 'geometry.json'), read(BASE / 'hardware-off.json')
    assert geometry['passed'] and hardware['passed']
    analysis = dict(passed=True, core=proof['core'], data=proof['data'], instructions=proof['instructions'],
        geometry=geometry, hardware_off=hardware, model_qualified=False, performance_qualified=False,
        production_changed=False, **resources)
    common.close(BASE, analysis, proof['files'], resources['identities'])


if __name__ == '__main__':
    main()
