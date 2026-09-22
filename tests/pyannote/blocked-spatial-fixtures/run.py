"""Capture actual layer operands under the existing bounded local supervisor."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/pyannote-blocked-spatial-fixtures-20260922'
MONITOR = ROOT/'tests/parakeet/packing-budgets/common.py'
module = importlib.util.spec_from_file_location('fixture_monitor', MONITOR)
monitor = importlib.util.module_from_spec(module); module.loader.exec_module(monitor)
monitor.BASE = BASE
pin, read, save, verify = monitor.pin, monitor.read, monitor.save, monitor.verify


def main():
    assert not BASE.exists()
    amd = ROOT/'artifacts/pyannote-blocked-spatial-amd-20260922'
    assert pin(amd/'closed.json')['sha256'] == 'f2ac8145a31e9d4801bca1ac0e13d18825ea87c409eb8bae4ca15dad19445c92'
    proof = read(amd/'closed.json'); assert proof['passed']
    for name, wanted in proof['files'].items(): assert pin(amd/name) == wanted, name
    BASE.mkdir(); (BASE/'logs').mkdir(); (BASE/'output').mkdir(); (BASE/'tools').mkdir()
    for p in TOOLS.iterdir():
        if p.is_file(): shutil.copy2(p, BASE/'tools'/p.name)
    shutil.copy2(ROOT/'.agent/m13-pyannote-blocked-spatial-20260922.md', BASE/'prospective-plan.md')
    native = ROOT/'artifacts/pyannote-native-layout-amd-20260922'
    census = ROOT/'artifacts/pyannote-blocked-spatial-census-20260922'
    paths = [MONITOR, amd/'closed.json', census/'closed.json', census/'census.json',
             native/'closed.json', native/'embedding-graph-and-execution.json', native/'payload/payload.json',
             ROOT/'models/pyannote-embedding/embedding_encoder.onnx', Path(sys.executable),
             BASE/'prospective-plan.md', *[p for p in TOOLS.iterdir() if p.is_file()],
             *[p for p in (BASE/'tools').iterdir() if p.is_file()]]
    for case in read(native/'payload/payload.json')['cases']:
        if case['model'] == 'embedding':
            for key in ['input', 'reference']:
                p = native/'payload'/case[key]; assert pin(p) == case[key+'_pin']; paths.append(p)
    packages = {}
    for name in ['numpy', 'onnx', 'onnxruntime', 'google.protobuf', 'psutil']:
        origin = Path(importlib.util.find_spec(name).origin).resolve(); packages[name] = str(origin)
        paths.extend(p for p in origin.parent.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files = {p.as_posix(): pin(p) for p in paths}
    save(BASE/'inputs.json', dict(files=files, packages=packages, no_performance_measurement=True))
    own = monitor.psutil.Process()
    state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    target = BASE/'controller.json'; save(target, state)
    try:
        monitor.worker(state, target, 'capture', [sys.executable, '-X', 'utf8', '-B', TOOLS/'capture.py'],
                       ROOT, [0], 12, 8, 900, False, BASE/'output')
        verify(files); result = read(BASE/'output/result.json'); assert result['passed']
        save(BASE/'verified.json', dict(passed=True, result=pin(BASE/'output/result.json'), files=files))
        state['code'] = 0
        print(json.dumps(dict(passed=True, calls=len(result['calls']), bytes=result['tensor_bytes'], arrays=result['arrays'])))
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); raise
    finally:
        state['complete'] = True; save(target, state)


if __name__ == '__main__': main()
