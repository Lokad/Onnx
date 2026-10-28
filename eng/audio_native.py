"""Independent pinned ORT applications; numerical checks stay outside API timing."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import time

from audio_protocol import check_result, pin, read, validate_records, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'manifest', 'output']:
        parser.add_argument(name, type=Path)
    parser.add_argument('mode', choices=['conformance', 'timing'])
    args = parser.parse_args()
    assert os.name == 'posix' and os.sched_getaffinity(0) == {2}
    assert not args.output.exists()
    manifest, root = read(args.manifest), args.root.resolve()
    family = manifest['family']
    assert manifest['schema'] == 1 and family in ('parakeet', 'pyannote')
    flags = {k: v for k, v in os.environ.items() if k.lower().startswith(('lokad_', 'dotnet_', 'complus_', 'omp_', 'mkl_', 'openblas_', 'blis_', 'numexpr_'))}
    assert not any(k.lower().startswith(('lokad_', 'dotnet_', 'complus_')) for k in flags)
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        assert os.environ.get(key) == '1'
    for name, version in manifest['native_versions'].items():
        assert importlib.metadata.version(name) == version, name

    def verify(spec):
        path = root/spec['path']
        assert pin(path) == {k: spec[k] for k in ['bytes', 'sha256']}, str(path)
        return path

    for group in ['models', 'native_sources', 'native_binaries']:
        for spec in manifest[group].values():
            verify(spec)
    verify(manifest['reference'])
    adapter = verify(manifest['adapter'])
    spec = importlib.util.spec_from_file_location('pinned_audio_adapter', adapter)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import numpy as np
    import onnxruntime as ort
    assert np.__version__ == '2.2.4' and ort.__version__ == '1.29.0'
    if family == 'pyannote':
        from torchaudio.compliance import kaldi
        assert hashlib.sha256(Path(inspect.getfile(kaldi)).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == manifest['pins']['kaldi_lf_sha256']
    cases = []
    for case in manifest['cases']:
        pcm = np.load(verify(case['pcm']), allow_pickle=False)
        assert pcm.dtype == np.float32 and pcm.shape == (case['samples'],) and np.isfinite(pcm).all()
        before = pcm.tobytes()
        assert hashlib.sha256(before).hexdigest() == case['raw_sha256']
        cases.append((case, pcm, before))
    args.output.mkdir()
    start = time.perf_counter()
    model = getattr(module, {'parakeet': 'Parakeet', 'pyannote': 'Pyannote'}[family])(root, manifest)
    setup = time.perf_counter()-start
    records, held, first = [], [], {}
    warmups = manifest['warmup_passes']
    for iteration in range(1 if args.mode == 'conformance' else warmups+manifest['measured_passes']):
        for case, pcm, before in cases:
            assert pcm.tobytes() == before
            for result, snapshot in held:
                assert json.dumps(result, sort_keys=True, allow_nan=False) == snapshot
            start = time.perf_counter_ns()
            actual = model(pcm)
            end = time.perf_counter_ns()
            error = check_result(actual, case['expected'], family=family)
            assert pcm.tobytes() == before
            snapshot = json.dumps(actual, sort_keys=True, allow_nan=False)
            if case['name'] in first:
                assert snapshot == first[case['name']]
            else:
                first[case['name']] = snapshot
            held.append((actual, snapshot))
            row = dict(name=case['name'], **{'pass': iteration}, phase='warmup' if args.mode == 'conformance' or iteration < warmups else 'measured',
                       seconds=(end-start)/1e9, start_ticks=start, end_ticks=end, frequency=1000000000,
                       result=actual, maximum_centroid_error=error, input_sha256=hashlib.sha256(before).hexdigest(), ownership=True)
            records.append(row)
            write(args.output/f'{len(records)-1:03d}.json', row)
            print(family, row['phase'], iteration, case['name'], row['seconds'], flush=True)
    assert all(pcm.tobytes() == before for _, pcm, before in cases)
    assert all(json.dumps(result, sort_keys=True, allow_nan=False) == snapshot for result, snapshot in held)
    import psutil
    numeric = {}
    for item in psutil.Process().memory_maps():
        path = Path(item.path)
        if path.is_file() and '.so' in path.name and any(token in str(path).lower() for token in ['numpy', 'scipy', 'torch', 'onnxruntime', 'blas', 'mkl']):
            numeric[str(path)] = pin(path)
    assert numeric and any('onnxruntime' in path for path in numeric)
    value = dict(schema=1, producer='audio-full-api-v1', family=family, engine='ort', conformance=args.mode == 'conformance',
                 records=records, setup_seconds=setup, manifest_sha256=pin(args.manifest)['sha256'], runner_sha256=pin(Path(__file__))['sha256'],
                 adapter_sha256=pin(adapter)['sha256'], runtime=sys.version, python_binary=pin(Path(sys.executable)), versions=manifest['native_versions'],
                 native_binaries=manifest['native_binaries'], numeric_libraries=numeric, affinity=4, flags=flags, held_outputs_unchanged=True,
                 native_settings=dict(provider='CPUExecutionProvider', intra_threads=1, inter_threads=1, sequential=True, graph_optimizations='all', spinning=False))
    validate_records(value, manifest, args.mode)
    write(args.output/'result.json', value)


if __name__ == '__main__':
    main()
