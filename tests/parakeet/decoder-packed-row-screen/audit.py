"""Reuse the qualified collection/build auditor, then score this fixed screen."""
import importlib.util
import json
from pathlib import Path
import sys
from run import BASE, REMOTE, TOOLS, pin, read, write
from score import ORDER, score

loader = importlib.util.spec_from_file_location('prior_screen_audit', TOOLS.parent/'vector-sigmoid-screen/audit.py')
prior = importlib.util.module_from_spec(loader); loader.loader.exec_module(prior)


def capture():
    assert not (BASE/'closed.json').exists()
    folder, spec, receipt, state, built, resources = prior.collected('capture')
    assert pin(folder/'build-review.json') == pin(BASE/'build-review.json')
    assert read(BASE/'build-review.json')['built'] == pin(folder/'built.json')
    assert read(BASE/'build-collected/build-state.json')['ended'] < state['started']
    loader = importlib.util.spec_from_file_location('screen_accounting', folder/'campaign_processes.py')
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
        reports[name] = value
    verdict = score(reports, read(folder/'census.json'))
    analysis = dict(passed=True, **verdict, resources=resources, products=spec['products'], consumer=built['consumer'],
        evidence=spec['evidence'], diagnostic_only=True, release_admitted=False, no_model_execution=True,
        no_application_score=True, foreign_cpu=[r['accounting'] for r in state['runs']],
        outputs={name: pin(folder/'logs'/(name+'.json')) for name in ORDER})
    write(BASE/'analysis.json', analysis)
    write(BASE/'closed.json', dict(passed=True, admitted=analysis['admitted'], analysis=pin(BASE/'analysis.json'),
        collection=pin(folder/'capture-collection.json'), transfer=pin(BASE/'capture-transfer.json'),
        terminal_owners=receipt['identities'], reviewer=pin(Path(__file__)),
        files={p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file()}))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'), admitted=analysis['admitted'], rows=analysis['rows'],
        failed_controls=[r for r in analysis['controls'] if not r['passed']], gates=analysis['gates'], resources=resources)))


if __name__ == '__main__': {'build': prior.build, 'capture': capture}[sys.argv[1]]()
