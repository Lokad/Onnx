import traceback
from common import *


def main():
    prepared = read(BASE / 'prepared.json')
    assert prepared['passed']
    verify(prepared['files'])
    built = read(BASE / 'build-state.json')
    assert built['complete'] and built['code'] == 0
    terminal(built['supervisor'])
    for row in built['runs']:
        for pid, birth in row['members'].items():
            terminal(dict(pid=int(pid), birth=birth))
    assert not (BASE / 'processes.json').exists()
    own = psutil.Process()
    state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    try:
        monitor.worker(state, BASE / 'processes.json', 'arithmetic',
                       ['dotnet', BASE / 'source/bin/Release/net10.0/Probe.dll', BASE / 'prepared.json', BASE / 'output'],
                       ROOT, [0], 4, 2, 600, False, BASE / 'output')
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
