"""Capture selected recurrent operands, then check every output with native ORT."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/pyannote-lstm-input-fixtures-v3-20260922'
SELECTED = ROOT/'artifacts/pyannote-blocked-spatial-models-20260922'
RUNTIME = ROOT/'artifacts/pyannote-blocked-spatial-composition-v3-20260922/runtime'
NATIVE = ROOT/'artifacts/pyannote-native-layout-amd-20260922'
MODEL = ROOT/'models/pyannote-segmentation/segmentation/model.onnx'
FEED = ROOT/'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
MONITOR = ROOT/'tests/parakeet/packing-budgets/common.py'
CORE = '3c2f16b08856426d3dfeff07f1638dd76cee7f06b65bbee230e8e0789679206f'
DATA = '6318cf48691470b908eec4c4d09c558172e43ce3b04bca9039c68966998a684b'
spec = importlib.util.spec_from_file_location('lstm_fixture_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec); spec.loader.exec_module(monitor); monitor.BASE = BASE
pin, read, save, verify = monitor.pin, monitor.read, monitor.save, monitor.verify
JOBS = {'restore': (8, 8, 900, False), 'build': (8, 8, 900, False), 'capture': (12, 8, 900, True), 'native': (12, 8, 900, True)}


def main():
    assert not BASE.exists()
    failed = ROOT/'artifacts/pyannote-lstm-input-fixtures-20260922'
    assert pin(failed/'failure-closed.json')['sha256'] == 'bd9198a67102f0875fe27d1705d8d04253b7eaa2030e54b15f89395d7d4885c2'
    failure = read(failed/'failure-closed.json'); assert failure['closed'] and not failure['passed'] and failure['no_model_execution']
    for name,wanted in failure['files'].items(): assert pin(failed/name) == wanted,name
    for identity in failure['identities']: monitor.terminal(identity)
    failed_v2 = ROOT/'artifacts/pyannote-lstm-input-fixtures-v2-20260922'
    assert pin(failed_v2/'failure-closed.json')['sha256'] == 'e8a978db45c7a8da6dd53a71703fd5e27a12c99aa8c8839578050fe0e36581a6'
    failure_v2 = read(failed_v2/'failure-closed.json'); assert failure_v2['closed'] and not failure_v2['passed']
    for name,wanted in failure_v2['files'].items(): assert pin(failed_v2/name) == wanted,name
    for identity in failure_v2['identities']: monitor.terminal(identity)
    previous = read(SELECTED/'closed.json'); assert previous['passed']
    verify(previous['files'])
    for identity in previous['terminal_identities']: monitor.terminal(identity)
    assert pin(RUNTIME/'Lokad.Onnx.dll')['sha256'] == CORE and pin(RUNTIME/'Lokad.Onnx.Data.dll')['sha256'] == DATA
    assert pin(MODEL)['sha256'] == 'af62796adfc46ab36fb27c183e7fe6530a745c665b64e949acd73bf01a18a31a'
    old = read(SELECTED/'output/result.json'); assert old['passed'] and old['core_sha256'] == CORE
    native = read(NATIVE/'payload/payload.json')
    cases = []
    paths = [failed_v2/'failure-closed.json', failed/'failure-closed.json', MONITOR, Path(sys.executable), SELECTED/'closed.json', SELECTED/'output/result.json', NATIVE/'payload/payload.json', MODEL,
             ROOT/'tests/parakeet/portable-models/common.py']
    for row in [r for r in native['cases'] if r['model'] == 'segmentation']:
        input_path = NATIVE/'payload'/row['input']; reference = NATIVE/'payload'/row['reference']
        assert pin(input_path) == row['input_pin'] and pin(reference) == row['reference_pin']
        retained, = [v for v in old['rows'] if v['name'] == row['name'] and v['model'] == 'segmentation' and v['pass'] == 0]
        selected = SELECTED/'output'/retained['output']['file']
        assert pin(selected)['sha256'] == retained['output']['sha256']
        cases.append(dict(name=row['name'], input=str(input_path), input_sha256=pin(input_path)['sha256'], selected=str(selected),
            selected_sha256=pin(selected)['sha256'], native=str(reference), shape=row['shape']))
        paths += [input_path, reference, selected]
    assert len(cases) == 3
    BASE.mkdir(); (BASE/'logs').mkdir(); (BASE/'consumer').mkdir()
    shutil.copytree(RUNTIME, BASE/'runtime')
    for name in ['Capture.cs', 'Capture.csproj']: shutil.copy2(TOOLS/name, BASE/'consumer'/name)
    shutil.copy2(ROOT/'PLAN.md', BASE/'prospective-plan.md')
    save(BASE/'capture-spec.json', dict(model=str(MODEL), model_sha256=pin(MODEL)['sha256'], core=CORE, data=DATA, cases=cases,
        maximum_tensor_bytes=128*1024**2, no_performance_measurement=True))
    paths += [BASE/'capture-spec.json', BASE/'prospective-plan.md']
    for folder in [TOOLS, BASE/'runtime', BASE/'consumer']: paths += [p for p in folder.iterdir() if p.is_file()]
    packages = {}
    for name in ['numpy', 'onnx', 'onnxruntime', 'google.protobuf', 'psutil']:
        origin = Path(importlib.util.find_spec(name).origin).resolve(); packages[name] = str(origin)
        paths += [p for p in origin.parent.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files = {p.as_posix(): pin(p) for p in paths}
    save(BASE/'inputs.json', dict(files=files, packages=packages))
    own = monitor.psutil.Process(); state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    path = BASE/'controller.json'; save(path, state)
    flags = monitor.FLAGS+['-p:NuGetAudit=false']; project = BASE/'consumer/Capture.csproj'
    commands = {
        'restore': ['dotnet', 'restore', project, *flags, '--source', FEED, '--packages', BASE/'packages'],
        'build': ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'],
        'capture': ['dotnet', BASE/'consumer/bin/Release/net10.0/LstmCapture.dll', BASE/'capture-spec.json', BASE/'output'],
        'native': [sys.executable, '-X', 'utf8', '-B', TOOLS/'native.py']}
    try:
        for name, job in JOBS.items():
            if name == 'capture':
                built = BASE/'consumer/bin/Release/net10.0'
                assert pin(built/'Lokad.Onnx.dll')['sha256'] == CORE and pin(built/'Lokad.Onnx.Data.dll')['sha256'] == DATA
                save(BASE/'binaries.json', dict(files={p.as_posix(): pin(p) for p in built.iterdir() if p.is_file()}))
            monitor.worker(state, path, name, commands[name], ROOT, [0], job[0], job[1], job[2], not job[3],
                BASE/('native' if name == 'native' else 'output' if name == 'capture' else 'consumer'))
            print(name, 'passed', flush=True)
        verify(files); verify(read(BASE/'binaries.json')['files'])
        assert read(BASE/'output/result.json')['passed'] and read(BASE/'native/result.json')['passed']
        save(BASE/'verified.json', dict(passed=True, capture=pin(BASE/'output/result.json'), native=pin(BASE/'native/result.json')))
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); raise
    finally:
        state['complete'] = True; save(path, state)


if __name__ == '__main__': main()
