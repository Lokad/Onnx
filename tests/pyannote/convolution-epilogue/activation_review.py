"""Select a remaining voice-branch piece using closed metadata, without inference."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'artifacts/pyannote-activation-review-20260921'
sys.path.insert(0, str(ROOT / 'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
import psutil


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')


def main():
    assert not BASE.exists()
    own = psutil.Process()
    original = own.cpu_affinity()
    own.cpu_affinity([0])
    try:
        trusted, files = {}, {}
        for relative, sha in [
            ('artifacts/pyannote-convolution-epilogue-20260921/closed.json', 'aaf2a6457d49f087b536b1966427ce1ab580554a568652f79c23a40b85ea4b84'),
            ('artifacts/pyannote-performance-profile-20260921/closed.json', '4b7542a8917e558c92ba632fada6830ddf8c870fccf49064dae17ff1de194d5c'),
            ('artifacts/pyannote-sampled-thread-time-20260921/model-closed.json', '71c2e9c36f31216fe538454426aea9a15d7beaa7a19f72c038aaa7fec6bf818f'),
            ('artifacts/pyannote-vector-bias-20260921/closed.json', '1b0b89a3e0a00b3cf4e47d5d2f00bd1596fc82a263244b77eb547a1ffbb74911')]:
            path = ROOT / relative
            assert pin(path)['sha256'] == sha
            proof = read(path)
            assert proof['passed']
            trusted.update(proof['files'])
            files[relative] = pin(path)

        def checked(relative):
            path = ROOT / relative
            assert pin(path) == trusted[relative], relative
            files[relative] = pin(path)
            return path

        census = read(checked('artifacts/pyannote-convolution-epilogue-20260921/analysis.json'))
        profile = read(checked('artifacts/pyannote-performance-profile-20260921/output/result.json'))
        trace = read(checked('artifacts/pyannote-sampled-thread-time-20260921/model-analysis.json'))
        cases = []
        for case in census['cases']:
            row, = [r for r in profile['rows'] if r['name'] == case['name'] and r['model'] == 'embedding' and r['phase'] == 'wall']
            operators = {n['name']: n['op'] for n in row['nodes']}
            assert len(operators) == len(row['nodes'])
            selected = [r for r in case['rows'] if operators[r['name']] == 'ConvRelu']
            assert len(selected) == 17 and sum(r['output_values'] for r in selected) == 20129280
            assert sum(operators[r['name']] == 'Conv' for r in case['rows']) == 19
            cases.append(dict(name=case['name'], convolutions=36, fused_convolutions=17,
                activation_values=20129280, output_payload_bytes=80517120,
                selected=[dict(name=r['name'], output_shape=r['output_shape'], values=r['output_values']) for r in selected]))
        diagnostics = []
        for capture in trace['diagnostics']:
            total = capture['selected_seconds']['dialogue-30s']
            selected = [r for r in capture['exclusive'] if r['marker'] == 'dialogue-30s'
                        and 'CPUExecutionProvider.ConvRelu(' in r['method']]
            value = sum(r['seconds'] for r in selected)
            diagnostics.append(dict(name=capture['name'], selected_thread_seconds=total,
                convrelu_caller_exclusive_thread_seconds=value, fraction=value / total,
                scope='Caller sampled managed thread weight, including possible inlined helpers; not isolated activation time or CPU time.'))
        source_files = [checked('artifacts/pyannote-vector-bias-20260921/source/src/Lokad.Onnx/' + name)
            for name in ['CPUExecutionProvider.Fusion.cs', 'GraphFusion.cs', 'TensorOps.Elementwise.cs', 'TensorOps.ConvPool.cs']]
        source = source_files[0].read_text(encoding='utf8')
        assert 'return FinishRelu(op, inner.Outputs[0], options);' in source
        assert 'Tensor<float>.ReluSpanFloat(df.Buffer.Span, df.Buffer.Span);' in source
        commit = 'e35653a8edee630762550103aa5e8d3356b71d56'
        object_name = commit + ':src/Lokad.Onnx/TensorOps.ConvPool.cs'
        content = subprocess.check_output(['git', 'show', object_name], cwd=ROOT)
        assert b'os[outRow + j] = fuseRelu && v < 0f ? 0f : v;' in content
        BASE.mkdir()
        target = BASE / 'voice-TensorOps.ConvPool.cs'
        target.write_bytes(content)
        files[target.relative_to(ROOT).as_posix()] = pin(target)
        analysis = dict(passed=True, inference_executed=False, cases=cases, sampled_callers=diagnostics,
            voice_commit=commit, voice_source=pin(target),
            scope='Static selective-import review. Activation fusion is a later optional piece; no implementation or latency gain.',
            supervisor=dict(pid=own.pid, birth=own.create_time()), affinity=own.cpu_affinity())
        save(BASE / 'analysis.json', analysis)
        for path in [Path(__file__), BASE / 'analysis.json']:
            files[path.relative_to(ROOT).as_posix()] = pin(path)
        save(BASE / 'closed.json', dict(passed=True, files=files, analysis=pin(BASE / 'analysis.json')))
        print(json.dumps(dict(closure=pin(BASE / 'closed.json'), sampled_callers=diagnostics)))
    finally:
        own.cpu_affinity(original)


if __name__ == '__main__':
    main()
