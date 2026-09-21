from prepare import *

if __name__ == '__main__':
    assert not (BASE / 'processes.json').exists()
    spec = read(BASE / 'prepared.json')
    verify(spec['files'])
    preparation = read(BASE / 'preparation.json')
    assert preparation['complete'] and preparation['code'] == 0
    terminal(preparation['supervisor'])
    owner = psutil.Process()
    state = dict(complete=False, code=None, supervisor=dict(pid=owner.pid, birth=owner.create_time()), runs=[])
    try:
        for name in spec['jobs']:
            verify(spec['files'])
            monitor.worker(state, BASE / 'processes.json', name,
                ['dotnet', BASE / 'runtime/Probe.dll', ROOT, BASE / 'manifest.json', BASE / name, name],
                ROOT, [0], 10, 4, 900, False, BASE / name)
            assert read(BASE / name / 'result.json')['passed']
            print(name, 'complete', flush=True)
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc())
        raise
    finally:
        state['complete'] = True
        save(BASE / 'processes.json', state)
