import traceback
from common import *


def main():
    prepared = read(BASE / 'prepared.json')
    assert prepared['passed']
    verify(prepared['files'])
    build = read(BASE / 'build-state.json')
    assert build['complete'] and build['code'] == 0
    terminal(build['supervisor'])
    for run in build['runs']:
        for pid, birth in run['members'].items():
            terminal(dict(pid=int(pid), birth=birth))
    assert not (BASE / 'processes.json').exists()
    own = psutil.Process()
    state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    runtime = BASE / 'runtime'
    try:
        folder = BASE / 'native'
        folder.mkdir()
        row = monitor.worker(state, BASE / 'processes.json', 'native',
            ['dotnet', runtime / 'TranscribeReplay.dll', ROOT / 'models/parakeet-tdt-0.6b-v3', REFERENCE, folder / 'result.json'],
            ROOT, [0, 1], 10, 8, 1200, False, folder)
        result = read(folder / 'result.json')
        assert result['core_sha256'] == prepared['core']['sha256'] and result['data_sha256'] == prepared['data']['sha256']
        assert result['application_passed'] and not result['errors'] and result['comparisons'] == 784 and result['values_compared'] == 3090494
        assert row['code'] == (0 if result['passed'] else 1)
        row['native_numeric_passed'] = result['passed']
        save(BASE / 'processes.json', state)
        print('native complete, numeric_passed', result['passed'], 'max_error', result['max_error'], flush=True)
        folder = BASE / 'public'
        folder.mkdir()
        monitor.worker(state, BASE / 'processes.json', 'public',
            ['dotnet', runtime / 'AudioBenchmark.dll', ROOT, CORPUS, folder / 'output', 'conformance'],
            ROOT, [0], 14, 12, 1200, False, folder)
        result = read(folder / 'output/result.json')
        assert result['core_sha256'] == prepared['core']['sha256'] and result['data_sha256'] == prepared['data']['sha256']
        assert len(result['records']) == 20 and result['held_outputs_unchanged']
        verify(prepared['files'])
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc())
        raise
    finally:
        state['complete'] = True
        save(BASE / 'processes.json', state)


if __name__ == '__main__':
    main()
