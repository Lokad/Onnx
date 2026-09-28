"""Bound one offline address extraction using the already qualified export monitor."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
sys.path.insert(0, '/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil

BASE = Path(__file__).resolve().parent
spec = json.loads((BASE/'spec.json').read_text())
loader = importlib.util.spec_from_file_location('original_export_monitor', spec['monitor'])
monitor = importlib.util.module_from_spec(loader); loader.loader.exec_module(monitor)
monitor.BASE = monitor.ROOT = BASE
monitor.spec = spec
monitor.LIMITS = dict(artifacts=512*1024**2)
read, pin, save, live = monitor.read, monitor.pin, monitor.save, monitor.live


def verify():
    assert psutil.boot_time() == spec['boot']
    for name, wanted in spec['files'].items():
        assert pin(BASE/name) == wanted, name
    for name, wanted in spec['external'].items():
        assert pin(name) == wanted, name


def main():
    assert not (BASE/'state.json').exists()
    own = psutil.Process(); own.cpu_affinity([0])
    monitor.idle(); verify(); monitor.preflight()
    (BASE/'logs').mkdir(); (BASE/'tmp').mkdir()
    state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()),
        runs=[], started=time.time(), inference_calls=0)
    save(BASE/'state.json', state)
    try:
        monitor.export(state, 'addresses', ['/usr/bin/pwsh', '-NoLogo', '-NoProfile', '-NonInteractive',
            '-File', BASE/'Read-Addresses.ps1', spec['trace'], spec['libraries'], BASE/'output'])
        verify()
        assert read(BASE/'output/summary.json')['passed']
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); traceback.print_exc()
    finally:
        state.update(complete=True, ended=time.time()); save(BASE/'state.json', state)
    return state['code']


if __name__ == '__main__':
    raise SystemExit(main())
