"""Retain complete outputs of the existing independently qualified native adapter."""
from pathlib import Path
import argparse
import ctypes
import hashlib
import importlib.metadata
import inspect
import json
import os
import sys
import time
import wave
from common import pin, read, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('artifact', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('mode', choices=['inputs', 'run'])
    a = p.parse_args()
    assert os.name == 'nt', 'The pinned native environment is Windows'
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    process, system = ctypes.c_size_t(), ctypes.c_size_t()
    assert kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(), ctypes.byref(process), ctypes.byref(system)) and process.value == 4
    root, base = a.root.resolve(), a.artifact.resolve()
    manifest = read(base / 'manifest.json')
    for name, version in manifest['pins']['versions'].items():
        assert importlib.metadata.version(name) == version, name
    for item in list(manifest['models'].values()) + list(manifest['native_assets'].values()) + list(manifest['upstream'].values()) + list(manifest['native_sources'].values()):
        assert pin(root / item['path']) == {k: item[k] for k in ['bytes', 'sha256']}, item['path']
    import numpy as np
    import onnxruntime as ort
    from torchaudio.compliance import kaldi
    assert hashlib.sha256(Path(inspect.getfile(kaldi)).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == manifest['pins']['kaldi_lf_sha256']
    sys.path.insert(0, str(root / 'tests/audio/comparison'))
    from native_adapters import Pyannote
    cases = []
    for c in manifest['cases']:
        path = base / 'inputs' / c['path']
        assert pin(path) == c['wav']
        with wave.open(str(path), 'rb') as source:
            assert (source.getnchannels(), source.getsampwidth(), source.getframerate()) == (1, 2, 16000)
            pcm = np.frombuffer(source.readframes(c['samples']), dtype='<i2').astype(np.float32) / np.float32(32768)
        assert pcm.shape == (c['samples'],) and hashlib.sha256(pcm.tobytes()).hexdigest() == c['pcm_sha256']
        cases.append((c, pcm, pcm.tobytes()))
    a.output.mkdir()
    if a.mode == 'inputs':
        write(a.output / 'inputs.json', dict(passed=True, cases=[dict(name=c['name'], samples=len(pcm), pcm_sha256=c['pcm_sha256']) for c, pcm, _ in cases], affinity=4))
        return
    start = time.perf_counter()
    model = Pyannote(root, manifest)
    setup = time.perf_counter() - start
    held, records = [], []
    for i, (c, pcm, before) in enumerate(cases):
        assert pcm.tobytes() == before
        assert all(json.dumps(result, sort_keys=True, allow_nan=False) == saved for result, saved in held)
        start = time.perf_counter_ns()
        actual = model(pcm)
        end = time.perf_counter_ns()
        assert pcm.tobytes() == before
        held.append((actual, json.dumps(actual, sort_keys=True, allow_nan=False)))
        row = dict(name=c['name'], seconds=(end-start)/1e9, start_ticks=start, end_ticks=end, frequency=1000000000,
                   result=actual, input_sha256=c['pcm_sha256'], ownership=True)
        records.append(row)
        write(a.output / f'{i:02d}.json', row)
        print('Complete', c['name'], row['seconds'], flush=True)
    assert all(pcm.tobytes() == before for _, pcm, before in cases)
    assert all(json.dumps(result, sort_keys=True, allow_nan=False) == saved for result, saved in held)
    binaries = {p.name: pin(p) for p in (Path(ort.__file__).parent / 'capi').iterdir() if p.suffix in ('.dll', '.pyd')}
    write(a.output / 'result.json', dict(schema=1, engine='ort', records=records, setup_seconds=setup, held_outputs_unchanged=True,
         manifest_sha256=pin(base / 'manifest.json')['sha256'], runner_sha256=pin(Path(__file__))['sha256'], adapter_sha256=pin(root / 'tests/audio/comparison/native_adapters.py')['sha256'],
         native_binaries=binaries, python=sys.version, python_binary=pin(sys.executable), versions=manifest['pins']['versions'], affinity=4,
         flags={k:v for k,v in os.environ.items() if k.startswith(('LOKAD_','DOTNET_','COMPlus_','OMP_','MKL_','OPENBLAS_'))},
         native_settings=dict(provider='CPUExecutionProvider', intra_threads=1, inter_threads=1, sequential=True, graph_optimizations='all', spinning=False)))


if __name__ == '__main__':
    main()
