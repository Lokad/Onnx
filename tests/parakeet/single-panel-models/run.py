"""Sequential complete-model requests on the exact composed binaries."""
import traceback
from common import *


def main():
    spec = read(BASE / 'prepared.json'); assert spec['passed']; verify(spec['files'])
    resources(BASE, 'build-state.json', {n: (8, 8, 900, False) for n in ['consumer-restore', 'consumer-build', 'consumer-instructions']})
    assert not (BASE / 'processes.json').exists()
    monitor.BASE = BASE
    own = psutil.Process()
    identity = dict(pid=own.pid, birth=own.create_time())
    state = dict(complete=False, code=None, supervisor=identity, runs=[])
    runtime = BASE / 'runtime'
    try:
        for name, args, minimum, rss in [
            ('native', ['dotnet', runtime / 'TranscribeReplay.dll', ROOT / 'models/parakeet-tdt-0.6b-v3', REFERENCE, BASE / 'native/result.json'], 10, 8),
            ('public', ['dotnet', runtime / 'AudioBenchmark.dll', ROOT, CORPUS, BASE / 'public/output', 'conformance'], 14, 12)]:
            (BASE / name).mkdir()
            monitor.worker(state, BASE / 'processes.json', name, args, ROOT, [0], minimum, rss, 1200, False, BASE / name)
            result = read(BASE / name / ('result.json' if name == 'native' else 'output/result.json'))
            assert result['core_sha256'] == CORE and result['data_sha256'] == DATA
            if name == 'native':
                assert result['passed'] and result['application_passed'] and not result['errors']
                assert result['comparisons'] == 784 and result['values_compared'] == 3090494
            else:
                assert len(result['records']) == 20 and result['held_outputs_unchanged']
            print(name, 'complete', flush=True)
        verify(spec['files']); state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); raise
    finally:
        state['complete'] = True; save(BASE / 'processes.json', state)
    pyannote = BASE / 'pyannote'; pyannote.mkdir(); (pyannote / 'logs').mkdir()
    save(pyannote / 'prepared.json', spec)
    monitor.BASE = pyannote
    state = dict(complete=False, code=None, supervisor=identity, runs=[])
    try:
        monitor.worker(state, pyannote / 'processes.json', 'pyannote',
            ['dotnet', runtime / 'GraphQualification.dll', ROOT, INPUT, pyannote / 'output', CORE],
            ROOT, [0], 10, 8, 900, False, pyannote / 'output')
        verify(spec['files']); state['code'] = 0
        print('pyannote complete', flush=True)
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); raise
    finally:
        state['complete'] = True; save(pyannote / 'processes.json', state)


if __name__ == '__main__': main()
