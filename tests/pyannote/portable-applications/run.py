"""Reuse qualified binaries; execute only the complete application checks."""
import shutil
import traceback
from common import *


def main():
    assert not BASE.exists()
    files = {}
    for path, sha in [(COMPLETE / 'closed.json', 'c07358dee34e3983ab7c212488772deaf0c973cc45274c5b505ac766e27753b4'),
        (PRIOR / 'closed.json', 'f687396b6a6966b80e41f100dbe3bdddd0c682c8d63bfc658744ce05f5d79a3d'),
        (TESTS / 'closed.json', 'bf9822426cd75a86cc8c02c065005f0005e1a80b4ee63d5c53313fed145235f6')]:
        if sha:
            assert pin(path)['sha256'] == sha
        proof = read(path)
        assert proof['passed']
        verify(proof['files'])
        for identity in proof['identities']:
            terminal(identity)
        files[rel(path)] = pin(path)
    assert read(TESTS / 'analysis.json')['products_unchanged']
    BASE.mkdir()
    (BASE / 'logs').mkdir()
    runtime = BASE / 'runtime'
    shutil.copytree(BUILD / 'runtime', runtime)
    for name in ['AudioBenchmark', 'NaturalMeetings']:
        for suffix in ['dll', 'deps.json', 'runtimeconfig.json']:
            shutil.copy2(PRIOR / 'application-runtime' / (name + '.' + suffix), runtime / (name + '.' + suffix))
    assert pin(runtime / 'Lokad.Onnx.dll')['sha256'] == CORE
    assert pin(runtime / 'Lokad.Onnx.Data.dll')['sha256'] == DATA
    assert pin(runtime / 'AudioBenchmark.dll')['sha256'] == '7eca033a1b986a4cb90621392639d230c95097cb703dd25274fd72d66c5ba4f1'
    assert pin(runtime / 'NaturalMeetings.dll')['sha256'] == '79e3e7990ba6aa29e42da788277aad41b774ff3b8c3966b18ab1101944d0c0f1'
    meetings = BASE / 'meetings'
    meetings.mkdir()
    shutil.copytree(PRIOR / 'meetings/inputs', meetings / 'inputs')
    manifest = read(PRIOR / 'meetings/manifest.json')
    manifest.update(core_sha256=CORE, data_sha256=DATA,
        accuracy_scope='Normal source/package build with LSTM admission guard; original complete consumer and native references; no new timing comparison.')
    save(meetings / 'manifest.json', manifest)
    for path in [*runtime.rglob('*'), *meetings.rglob('*'), *TOOLS.iterdir(), MONITOR, INPUT,
        ROOT / 'tests/audio/comparison/audit.py', ROOT / 'tests/pyannote/natural-meetings/audit.py',
        ROOT / 'tests/pyannote/natural-meetings/common.py',
        PRIOR / 'dialogue-output/result.json', PRIOR / 'meetings-run-output/result.json',
        MEETINGS / 'prior/native.json']:
        if path.is_file():
            files[rel(path)] = pin(path)
    # Bind external model and PCM dependencies even though inference only reads them.
    for value in [read(INPUT), manifest]:
        specs = list(value['models'].values()) + list(value.get('native_assets', {}).values())
        specs += [case['pcm'] for case in value['cases'] if 'pcm' in case]
        if 'reference' in value:
            specs.append(value['reference'])
        for item in specs:
            path = ROOT / item['path']
            assert pin(path) == {key: item[key] for key in ['bytes', 'sha256']}
            files[rel(path)] = pin(path)
    save(BASE / 'prepared.json', dict(passed=True, files=files, core=pin(runtime / 'Lokad.Onnx.dll'), data=pin(runtime / 'Lokad.Onnx.Data.dll'),
        scope='Complete public output/ownership/resource qualification; fixed integrated product and existing package; no new benchmark or AMD promotion.'))
    own = psutil.Process()
    state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    save(BASE / 'processes.json', state)
    try:
        jobs = [('dialogue', ['dotnet', runtime / 'AudioBenchmark.dll', ROOT, INPUT, BASE / 'dialogue-output', 'timing'], 900)]
        jobs += [('meetings-' + mode, ['dotnet', runtime / 'NaturalMeetings.dll', ROOT, meetings, BASE / ('meetings-' + mode + '-output'), mode], 3600)
            for mode in ['inputs', 'run']]
        for name, args, seconds in jobs:
            verify(files)
            monitor.worker(state, BASE / 'processes.json', name, args, ROOT, [0], 10, 8, seconds, False, BASE / (name + '-output'))
            print(name, 'passed', flush=True)
        verify(files)
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc())
        raise
    finally:
        state['complete'] = True
        save(BASE / 'processes.json', state)


if __name__ == '__main__':
    main()
