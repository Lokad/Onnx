"""Close the successful finite continuation independently of its timing verdict."""
from common import *

FINISH = ROOT / 'artifacts/pyannote-vector-bias-comparison-finish-20260921'


def main():
    assert not (FINISH / 'closed.json').exists()
    prepared, state = read(FINISH / 'prepared.json'), read(FINISH / 'state.json')
    verify(prepared['tools'])
    assert state['complete'] and state['code'] == 0 and state['phase'] == 'complete'
    assert [r['name'] for r in state['stages']] == prepared['stages']
    identities = [state['supervisor'], prepared['application']]
    for row in state['stages']:
        assert row['complete'] and row['code'] == 0
        assert row['seconds'] < (12000 if row['name'] == 'run-comparison' else 300)
        identities.extend(dict(pid=int(pid), birth=birth) for pid, birth in row['members'].items())
    for identity in identities:
        terminal(identity)
    assert state['application_closure'] == pin(QUALIFIED / 'closed.json')
    assert state['comparison_closure'] == pin(BASE / 'closed.json')
    application, comparison = read(QUALIFIED / 'closed.json'), read(BASE / 'closed.json')
    assert application['passed'] and comparison['passed']
    assert state['attribution_valid'] == comparison['attribution_valid'] == read(BASE / 'analysis.json')['attribution_valid']
    verify(comparison['files'])
    for name, wanted in comparison['external_files'].items():
        assert pin(Path(name)) == wanted
    files = dict(prepared['tools'])
    for path in [QUALIFIED / 'closed.json', BASE / 'closed.json', Path(__file__), *FINISH.iterdir()]:
        if path.is_file():
            files[path.relative_to(ROOT).as_posix()] = pin(path)
    save(FINISH / 'closed.json', dict(passed=True, files=files, terminal_identities=identities,
        attribution_valid=comparison['attribution_valid'],
        scope='Finite continuation completed; comparison accuracy/resource verdict and latency verdict remain separate.'))
    print(dict(closed=pin(FINISH / 'closed.json'), identities=len(identities), attribution_valid=comparison['attribution_valid']))


if __name__ == '__main__':
    main()
