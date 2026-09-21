"""Exercise PID attachment and retained stack export without a model."""
from common import *


def main():
    spec = read(BASE / 'prepared.json')
    verify_spec(spec)
    preparation = read(BASE / 'preparation.json')
    assert preparation['complete'] and preparation['code'] == 0
    terminal(preparation['supervisor'])
    assert not (BASE / 'toy-processes.json').exists()
    state = new_state()
    try:
        pair(state, BASE / 'toy-processes.json', 'toy',
            ['dotnet', BASE / 'toy-source/bin/Release/net10.0/Toy.dll', BASE / 'toy-output', 'sampled'], BASE / 'toy-output', True, 8)
        for format in ['Speedscope', 'Chromium']:
            monitor.worker(state, BASE / 'toy-processes.json', 'export-' + format.lower(),
                ['dotnet', BASE / 'tracer/dotnet-trace.dll', 'convert', BASE / 'toy-output/capture.nettrace',
                    '--format', format, '--output', BASE / 'toy-output' / format.lower()],
                ROOT, [0], 8, 8, 900, False, BASE / 'toy-output')
        verify_spec(spec)
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc())
        raise
    finally:
        state['complete'] = True
        save(BASE / 'toy-processes.json', state)


if __name__ == '__main__':
    main()
