"""Execute all prospective application workers without overlapping inference."""
import traceback
from common import *


def main():
    prepared = read(BASE / 'prepared.json')
    assert prepared['passed']
    verify_prepared(prepared)
    assert not (BASE / 'processes.json').exists()
    for identity in read(QUALIFIED / 'closed.json')['identities']:
        terminal(identity)
    own = psutil.Process()
    state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    auditor = public_auditor()
    manifest = manifest_with_raw_hashes()
    expected = {r['name']: r['result'] for r in read(QUALIFIED / 'dialogue-output/result.json')['records']}
    limits = prepared['limits']
    try:
        for index, role in enumerate(prepared['jobs']):
            verify_prepared(prepared)
            name = f'{index}-{role}'
            folder = BASE / name
            folder.mkdir()
            prefix = [sys.executable, '-X', 'utf8', '-B', NATIVE] if role == 'ort' else ['dotnet', BASE / ('runtime-' + role) / 'AudioBenchmark.dll']
            command = prefix + [ROOT, INPUT, folder / 'output', 'timing']
            row = monitor.worker(state, BASE / 'processes.json', name, command, ROOT, [0],
                limits['preflight_gib'], limits['rss_gib'], limits['seconds'], False, folder)
            result = read(folder / 'output/result.json')
            auditor.validate_worker(result, manifest, 'timing')
            if role == 'ort':
                assert result['native_binaries'] == prepared['native_binaries'] and result['native_settings'] == prepared['native_settings']
            else:
                for key, filename in [('core_sha256', 'Lokad.Onnx.dll'), ('data_sha256', 'Lokad.Onnx.Data.dll'), ('runner_sha256', 'AudioBenchmark.dll')]:
                    assert result[key] == prepared['roles'][role][filename]['sha256']
                assert all(r['result'] == expected[r['name']] for r in result['records'])
            terminal(row['worker'])
            verify_prepared(prepared)
            row['application_passed'] = True
            save(BASE / 'processes.json', state)
            print(name, '16 requests passed', flush=True)
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc())
        raise
    finally:
        state['complete'] = True
        save(BASE / 'processes.json', state)


if __name__ == '__main__':
    main()
