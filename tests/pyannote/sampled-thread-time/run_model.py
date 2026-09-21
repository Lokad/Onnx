"""Run one normal diagnostic control and two bounded sampled processes sequentially."""
import hashlib
from common import *
import numpy as np


def main():
    spec = read(BASE / 'model-prepared.json')
    verify_spec(spec)
    preparation = read(BASE / 'model-preparation.json')
    assert preparation['complete'] and preparation['code'] == 0
    terminal(preparation['supervisor'])
    assert not (BASE / 'model-processes.json').exists()
    auditor_spec = importlib.util.spec_from_file_location('original_public_auditor', ROOT / 'tests/audio/comparison/audit.py')
    auditor = importlib.util.module_from_spec(auditor_spec)
    auditor_spec.loader.exec_module(auditor)
    manifest = read(INPUT)
    for case in manifest['cases']:
        pcm = np.load(ROOT / case['pcm']['path'], allow_pickle=False)
        case['raw_sha256'] = hashlib.sha256(pcm.tobytes()).hexdigest()
    expected = {r['name']:r['result'] for r in read(QUALIFIED / 'dialogue-output/result.json')['records']}
    state = new_state()
    try:
        for name in spec['jobs']:
            verify_spec(spec)
            sampled = name != 'control'
            output = BASE / name
            pair(state, BASE / 'model-processes.json', name,
                ['dotnet', BASE / 'runtime/SampledAudio.dll', ROOT, INPUT, output, 'timing', 'sampled' if sampled else 'control'],
                output, sampled, 10)
            result = read(output / 'result.json')
            auditor.validate_worker(result, manifest, 'timing')
            assert result['sampled'] == sampled and result['core_sha256'] == CORE and result['data_sha256'] == DATA
            assert result['runner_sha256'] == spec['consumer']['sha256']
            assert all(row['result'] == expected[row['name']] for row in result['records'])
            assert read(output / 'ready.json')['warmup_records'] == 4
            print(name, '16 original public checks passed', flush=True)
        for name in spec['jobs'][1:]:
            for format in ['Speedscope', 'Chromium']:
                monitor.worker(state, BASE / 'model-processes.json', name+'-'+format.lower(),
                    ['dotnet', BASE / 'tracer/dotnet-trace.dll', 'convert', BASE / name / 'capture.nettrace',
                        '--format', format, '--output', BASE / name / format.lower()],
                    ROOT, [0], 8, 8, 900, False, BASE / name)
        verify_spec(spec)
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc())
        raise
    finally:
        state['complete'] = True
        save(BASE / 'model-processes.json', state)


if __name__ == '__main__':
    main()
