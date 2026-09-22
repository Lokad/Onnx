"""Export immutable AMD captures locally with the original pinned collector."""
import traceback
from common import *


def main():
    prepared()
    receipt = read(BASE / 'collected/collection.json')
    assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
    for name, wanted in receipt['files'].items(): assert pin(BASE / 'collected' / name) == wanted, name
    assert not (BASE / 'export-state.json').exists()
    state = new_state()
    try:
        for name in ['sampled-a','sampled-b']:
            output = BASE / 'exports' / name; output.mkdir(parents=True)
            for format in ['Speedscope','Chromium']:
                monitor.worker(state, BASE / 'export-state.json', name+'-'+format.lower(),
                    ['dotnet', BASE / 'payload/tracer/dotnet-trace.dll', 'convert', BASE / 'collected' / name / 'capture.nettrace',
                        '--format', format, '--output', output / format.lower()],
                    ROOT, [0], 8, 8, 900, False, output)
                print(name, format, 'exported', flush=True)
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); raise
    finally:
        state['complete'] = True; save(BASE / 'export-state.json', state)


if __name__ == '__main__': main()
