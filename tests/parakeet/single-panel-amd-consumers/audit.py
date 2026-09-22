"""Close only terminal, intact consumer qualification with every original case."""
import importlib.util
import checks
from prepare import ROOT, BASE, SHAPES, read, pin, verify, inspect, caller_source, probe_source, bridge_source


def main():
    assert not (BASE / 'closed.json').exists()
    proof = read(BASE / 'prepared.json'); assert proof['passed']
    verify(proof['files'])
    for relative, expected in [('caller/Program.cs', caller_source()), ('probe/Probe.cs', probe_source()), ('bridge/Program.cs', bridge_source())]:
        assert (BASE / relative).read_text(encoding='utf8') == expected
    instructions = inspect()
    reports = {}
    for mode in ['normal', 'disabled']:
        reports[mode] = dict(caller=checks.caller(read(BASE / ('caller-' + mode + '.json')), read(SHAPES), mode, '10.0.12'),
            probe=checks.probe(read(BASE / ('probe-' + mode + '.json')), mode, '10.0.12', False))
    spec = importlib.util.spec_from_file_location('consumer_resource_audit', ROOT / 'tests/pyannote/convolution-portable-qualification/common.py')
    common = importlib.util.module_from_spec(spec); spec.loader.exec_module(common)
    jobs = {name: (8, 900, name.startswith(('caller-normal', 'caller-disabled', 'probe-normal', 'probe-disabled')))
        for name in ['bridge-restore', 'bridge-build', 'caller-restore', 'caller-build', 'probe-restore', 'probe-build',
            'caller-instructions', 'caller-normal', 'probe-normal', 'caller-disabled', 'probe-disabled']}
    resources = common.resources(BASE, 'processes.json', jobs)
    state = read(BASE / 'processes.json')
    for mode in ['normal', 'disabled']:
        owner = next(r for r in state['runs'] if r['name'] == 'caller-' + mode)
        assert read(BASE / ('caller-' + mode + '.json'))['pid'] == owner['worker']['pid']
    analysis = dict(passed=True, instructions=instructions, checks=reports, **resources,
        production_changed=False, performance_qualified=False, amd_precedence_executed=False)
    common.close(BASE, analysis, proof['files'], resources['identities'])


if __name__ == '__main__': main()
