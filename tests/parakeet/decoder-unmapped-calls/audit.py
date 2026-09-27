"""Validate the unchanged capture scope plus every new individual interval."""
import importlib.util
import json
from pathlib import Path
import sys
from run import BASE, REMOTE, TOOLS, PREVIOUS, PARENT, pin, read, write
from observations import ORDER, analyze
from score import score

loader = importlib.util.spec_from_file_location('unchanged_build_audit', TOOLS.parent/'vector-sigmoid-screen/audit.py')
prior = importlib.util.module_from_spec(loader); loader.loader.exec_module(prior)


def capture():
    assert not (BASE/'closed.json').exists()
    folder, spec, receipt, state, built, resources = prior.collected('capture')
    assert pin(folder/'build-review.json') == pin(BASE/'build-review.json')
    assert read(BASE/'build-review.json')['built'] == pin(folder/'built.json')
    assert read(BASE/'build-collected/build-state.json')['ended'] < state['started']
    loader = importlib.util.spec_from_file_location('captured_accounting', folder/'campaign_processes.py')
    accounting = importlib.util.module_from_spec(loader); loader.loader.exec_module(accounting)
    reports = {}
    for sequence, run in enumerate(state['runs']):
        name = run['name']; role = name.split('-')[0]
        wanted = accounting.foreign_fraction(run['cpu_before'], run['cpu_after'], state['supervisor']['pid'])
        assert wanted == run['accounting'] and wanted['valid'] and wanted['foreign_cpu_fraction'] <= .01
        value = read(folder/'logs'/(name+'.json'))
        assert value['pid'] == run['owner']['pid'] and value['runtime'] == '10.0.8' and not value['flags']
        assert value['core_sha256'] == spec['products'][role]['sha256'] and value['assembly'] == built['consumer']['sha256']
        assert value['census_sha256'] == spec['census']['sha256']
        assert run['command'] == ['/home/vermorel/.dotnet/dotnet', REMOTE+'/runtime/'+role+'/Screen.dll', REMOTE, role, str(sequence), REMOTE+'/logs/'+name+'.json']
        original = read(PREVIOUS/'capture-collected/logs'/(name+'.json'))
        for old, new in zip(original['rows'], value['rows'], strict=True):
            assert all(old[k] == new[k] for k in ['input_sha256', 'weight_sha256', 'output_sha256']), new['name']
        reports[name] = value
    comparison = score(reports, read(folder/'census.json'))
    observations = analyze(reports)
    analysis = dict(passed=True, **observations, instrumented_comparison=comparison,
        products=spec['products'], consumer=built['consumer'], prior_screen=spec['prior_screen'],
        resources=resources, foreign_cpu=[r['accounting'] for r in state['runs']], diagnostic_only=True,
        evidence=spec['evidence'], outputs={n: pin(folder/'logs'/(n+'.json')) for n in ORDER})
    write(BASE/'analysis.json', analysis)
    write(BASE/'closed.json', dict(passed=True, admitted=False, release_admitted=False,
        analysis=pin(BASE/'analysis.json'), collection=pin(folder/'capture-collection.json'),
        terminal_owners=receipt['identities'], reviewer=pin(Path(__file__)),
        files={p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file()}))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'), first_call_explanation_supported=observations['first_call_explanation_supported'],
        checks=observations['checks'], batch_ratio=observations['batch_ratio'],
        positions=observations['positions'], excess_fraction=observations['first_call_excess_fraction'], resources=resources)))


if __name__ == '__main__': {'build': prior.build, 'capture': capture}[sys.argv[1]]()
