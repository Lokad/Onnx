"""Check exact pointwise shapes, source dispatch, and retained trace scope.

Reads existing evidence only. No inference, variant selection, or new score.
"""
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
ORT = '2e2543fbe9fae542f921d47a72d21d5a4ef0b710'


def pin(path):
    data = path.read_bytes()
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def git(directory, *args):
    return subprocess.check_output(
        ['git', '-c', 'gc.auto=0', '-C', str(directory), *args])


def main():
    native = ROOT/'artifacts/parakeet-decoder-lstm-layout-ort-profile-amd-20260927'
    assert pin(native/'closed.json')['sha256'] == '17c9be907e142700837ad86da026141882cfabb022fa9273e4c8bdd8971884d7'
    closed = read(native/'closed.json')
    assert closed['passed'] and closed['analysis'] == pin(native/'analysis.json')
    shapes = read(native/'analysis.json')['profiles']['encoder']['shapes']
    selected = [r for r in shapes if '/pointwise_conv' in r['name']]
    grouped = collections.defaultdict(list)
    for row in selected:
        grouped[row['name']].append(row)
    expected = {f'/layers.{i}/conv/pointwise_conv{j}/Conv'
                for i in range(24) for j in (1, 2)}
    assert set(grouped) == expected
    widths = None
    for name, rows in grouped.items():
        assert sum(r['calls'] for r in rows) == 80
        m = 2048 if 'pointwise_conv1' in name else 1024
        current = set()
        for row in rows:
            assert len(row['inputs']) == 2 and len(row['outputs']) == 1
            x, w = [v['float'] for v in row['inputs']]
            y = row['outputs'][0]['float']
            assert x[:2] == [1, 1024] and w == [m, 1024, 1]
            assert y == [1, m, x[2]]
            current.add(x[2])
        if widths is None:
            widths = current
        assert widths == current
    assert sorted(widths) == [51,61,83,88,89,102,106,112,114,120,151,156,157,158,167,169,190,222,225]
    assert all(t % 32 for t in widths)

    sources = {}
    for name in ['TensorOps.ConvPool.cs', 'TensorOps.MatMul.cs',
                 'MathOps.cs', 'MathOps.PackedAvx512.cs', 'DenseTensor.cs']:
        relative = 'src/Lokad.Onnx/' + name
        data = (ROOT/relative).read_bytes()
        # Working-tree CRLF is allowed; source content must match the release.
        assert data.replace(b'\r\n', b'\n') == git(ROOT, 'show', 'edc1a6c8:'+relative).replace(b'\r\n', b'\n')
        sources[relative] = pin(ROOT/relative)
    text = (ROOT/'src/Lokad.Onnx/MathOps.PackedAvx512.cs').read_text()
    assert 'k % 32 != 0' in text
    native_sources = {}
    for path in ['convolve.cpp', 'sgemm.cpp', 'mlasi.h',
                 'x86_64/FgemmKernelAvx512FCommon.h']:
        relative = 'onnxruntime/core/mlas/lib/' + path
        data = git(ROOT/'external/onnxruntime', 'show', ORT+':'+relative)
        native_sources[relative] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        if path == 'mlasi.h':
            assert '#define MLAS_SGEMM_STRIDEN                          128' in data.decode()
            assert '#define MLAS_SGEMM_STRIDEK                          128' in data.decode()
    blocking = []
    for t in sorted(widths):
        stride_n = stride_k = 128
        while stride_n > 16 and stride_n // 2 >= t:
            stride_n //= 2
            stride_k *= 2
        blocking.append(dict(output_columns=t, ort_stride_n=stride_n,
            ort_stride_k=stride_k, reduction_slices=1024//stride_k,
            column_slices=(t+stride_n-1)//stride_n,
            managed_full_32_column_panels=t//32, managed_tail_columns=t%32,
            existing_avx512_helper_accepts=False))

    # Existing exports can rule out reuse for node-specific attribution without
    # attributing their older sample weights to the current product.
    old = ROOT/'artifacts/parakeet-current-profile-amd-20260923'
    proof = read(old/'closed.json')
    assert proof['passed']
    parser_path = TOOLS.parent/'current-profile-amd/selected_stacks.py'
    spec = importlib.util.spec_from_file_location('retained_pointwise_scope', parser_path)
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    traces = []
    for role in ['sampled-a', 'sampled-b']:
        documents = []
        pins = {}
        for filename in ['speedscope.speedscope.json', 'chromium.chromium.json']:
            path = old/'exports'/role/filename
            relative = path.relative_to(ROOT).as_posix()
            assert pin(path) == proof['files'][relative]
            pins[relative] = pin(path)
            documents.append(read(path))
        cross = parser.cross_export(*documents)
        frames = [f['name'] for f in documents[0]['shared']['frames']]
        matches = [s for s in frames if 'RunPointwiseBatchesFloat(' in s]
        assert not matches
        assert any('Conv2DFloatCore(' in s for s in frames)
        assert any('mm_unsafe_vectorized_intrinsics_2x4packed_bump(' in s for s in frames)
        traces.append(dict(name=role, files=pins, cross_export=cross,
            pointwise_frames=matches, pointwise_costs_separately_identifiable=False))
    result = dict(passed=True, inference_calls=0, source=pin(Path(__file__)),
        native_profile=pin(native/'closed.json'), qualified_source='edc1a6c8',
        managed_sources=sources, ort_revision=ORT, ort_sources=native_sources,
        pointwise_nodes=len(grouped), calls_including_warmup=sum(r['calls'] for r in selected),
        blocking=blocking, retained_traces=traces,
        scope='Source-derived routes; no new per-node kernel observation or timing attribution.',
        next_observation='Separate materialization, clearing, packing and arithmetic for these 48 pointwise nodes.')
    with (TOOLS/'pointwise-source-observations-20260927.json').open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(passed=True, nodes=len(grouped), widths=len(widths),
        existing_avx512_eligible_widths=0, pointwise_frames_in_retained_traces=0,
        inference_calls=0)))


if __name__ == '__main__':
    main()
