"""Reuse the closed decoder trace to separate named projections from unresolved LSTM time."""
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-decoder-lstm-current-review-20260927'
TRACE = ROOT/'artifacts/parakeet-decoder-projection-observation-v3-amd-20260927'
QUALIFIED = ROOT/'artifacts/parakeet-rational-sigmoid-root-amd-20260927'
PREVIOUS = ROOT/'artifacts/parakeet-decoder-current-review-20260927'
REVISION = '2e2543fbe9fae542f921d47a72d21d5a4ef0b710'
RULES = ROOT/'tests/parakeet/decoder-projection-observation/events.py'
NATIVE = ['deep_cpu_lstm.cc', 'uni_directional_lstm.cc', 'rnn_helpers.cc']


def read(p): return json.loads(p.read_text(encoding='utf8'))


def pin(p):
    with p.open('rb') as f:
        return dict(bytes=p.stat().st_size, sha256=hashlib.file_digest(f, 'sha256').hexdigest())


def classify(stack, frames):
    names = '\n'.join(frames[i]['name'] for i in stack)
    if 'CPUExecutionProvider.LstmProjectOrdered(' in names: return 'lstm-ordered-projection'
    if 'CPUExecutionProvider.Lstm(' in names: return 'lstm-unresolved'
    if 'mm_m1_kblocked' in names: return 'vocabulary-row-major'
    return 'other' if stack else 'empty'


def analyze():
    inputs = {}
    for folder, digest in [(TRACE, 'ef94387e0517b08fd87d010c08d99e0b801e1bd7d2809c04b1dcec614333397d'),
                           (QUALIFIED, 'f6df53ce2ab773898bc84f144abeda97908d383fef9a4df4b7299a22a4d3594d'),
                           (PREVIOUS, '0faf5c988673b253edfdf07835f14db30b2dad43dd696e00224a99a0deb6a30f')]:
        assert pin(folder/'closed.json')['sha256'] == digest
        proof = read(folder/'closed.json'); assert proof['passed']
        assert proof['analysis'] == pin(folder/'analysis.json')
        inputs[(folder/'closed.json').relative_to(ROOT).as_posix()] = pin(folder/'closed.json')
        inputs[(folder/'analysis.json').relative_to(ROOT).as_posix()] = pin(folder/'analysis.json')
    proof = read(TRACE/'closed.json'); value = read(TRACE/'analysis.json')
    assert value['product'] == read(QUALIFIED/'analysis.json')['built']
    assert value['diagnostic_only'] and not value['performance_admitted']
    for name in ['trace-stacks/speedscope.speedscope.json', 'trace-capture/result.json']:
        path = TRACE/'collected'/name
        assert pin(path) == proof['files']['collected/'+name]
        inputs[path.relative_to(ROOT).as_posix()] = pin(path)
    assert pin(RULES) == read(TRACE/'prepared.json')['files'][RULES.relative_to(ROOT).as_posix()]
    inputs[RULES.relative_to(ROOT).as_posix()] = pin(RULES)
    loader = importlib.util.spec_from_file_location('retained_stack_accounting', RULES)
    rules = importlib.util.module_from_spec(loader); loader.loader.exec_module(rules)
    rules.classify = classify
    trace = read(TRACE/'collected/trace-capture/result.json')
    sampled = rules.stacks(read(TRACE/'collected/trace-stacks/speedscope.speedscope.json'),
                           value['events']['intervals'], trace['native_thread'])
    inside, leaves = defaultdict(float), defaultdict(float)
    for row in sampled['inside']:
        if row['iteration'] >= 256: inside[row['category']] += row['estimated_thread_ms']
    for row in sampled['leaves']:
        if row['iteration'] >= 256 and row['category'] == 'lstm-unresolved':
            leaves[row['frame']] += row['estimated_thread_ms']
    original = sum(r['estimated_thread_ms'] for r in value['stack_intervals']['inside'] if r['iteration'] >= 256)
    assert math.isclose(sum(inside.values()), original, rel_tol=0, abs_tol=1e-6)
    assert math.isclose(sum(leaves.values()), inside['lstm-unresolved'], rel_tol=0, abs_tol=1e-6)
    assert sampled['profiles'] == value['stack_intervals']['profiles']
    assert sampled['rounding_adjustments_ms'] == value['stack_intervals']['rounding_adjustments_ms']
    sources = {}
    for name in NATIVE:
        path = 'onnxruntime/core/providers/cpu/rnn/'+name
        process = subprocess.run(['git', '-c', 'gc.auto=0', '-C', str(ROOT/'external/onnxruntime'),
                                  'show', REVISION+':'+path], check=True, capture_output=True, timeout=30)
        sources[name] = process.stdout
    applied = read(QUALIFIED/'bundle/evidence/root-applied.json')
    for name in ['CPUExecutionProvider.Recurrent.cs', 'CPUExecutionProvider.LstmPanels.cs', 'GraphLstmPacking.cs']:
        path = ROOT/'src/Lokad.Onnx'/name
        assert pin(path) == applied['source_files'][path.relative_to(ROOT).as_posix()]
        inputs[path.relative_to(ROOT).as_posix()] = pin(path)
    panel = (ROOT/'src/Lokad.Onnx/CPUExecutionProvider.LstmPanels.cs').read_text()
    assert 'Vector.Add(a, Vector.Multiply(x, Vector.LoadUnsafe(ref weights, row)))' in panel
    assert '(k * output.Length + o)' in panel
    recurrent = (ROOT/'src/Lokad.Onnx/CPUExecutionProvider.Recurrent.cs').read_text()
    assert '"sigmoid" => v => 1f / (1f + MathF.Exp(-v))' in recurrent
    assert '"tanh" => MathF.Tanh' in recurrent
    previous = read(PREVIOUS/'analysis.json')
    shapes = [r for r in previous['observed_native_shapes'] if '/lstm/LSTM' in r['name']]
    assert len(shapes) == 2
    assert all(r['calls'] == 4800 and [i['float'] for i in r['inputs']] ==
               [[1, 1, 640], [1, 5120], [1, 1, 640], [1, 1, 640]] for r in shapes)
    lstm = inside['lstm-ordered-projection']+inside['lstm-unresolved']
    return dict(passed=True, new_inference_calls=0, product_changed=False, selected_candidate=None,
        inputs=inputs, product=value['product'], fixture_scope='One retained original decoder fixture; 1,024 post-warmup calls.',
        intervals_ms=dict(inside), total_observed_decoder_ms=original,
        projection_fraction_of_lstm=inside['lstm-ordered-projection']/lstm,
        unresolved_fraction_of_lstm=inside['lstm-unresolved']/lstm,
        unresolved_leaves_ms=dict(sorted(leaves.items(), key=lambda r: -r[1])),
        source_revision=REVISION,
        native_sources={name: dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest()) for name, data in sources.items()},
        prior_profile_lstm=next(r for r in previous['partition'] if r['group'] == 'Both recurrent nodes'),
        native_profile_shapes=shapes,
        managed_projection_geometry=dict(rows=1, reduction=640, outputs=2560, prepared_row_stride_bytes=10240,
                                         weight_bytes=6553600, separate_input_recurrent_outputs=True),
        per_node_native_kernel_proved=False, per_call_jit_version_proved=False, activation_time_measured=False,
        scope='Reclassified sampled-thread intervals, not kernel timers or a new score. Keep unresolved frames unresolved.'), sources


if __name__ == '__main__':
    assert sys.argv[1:] in [[], ['--publish']]
    value, sources = analyze(); destination = OUT/'observations-20260927.json'
    if sys.argv[1:]:
        assert not BASE.exists() and not destination.exists(); BASE.mkdir()
        (BASE/'native-source').mkdir()
        for name, data in sources.items(): (BASE/'native-source'/name).write_bytes(data)
        raw = json.dumps(value, indent=2, allow_nan=False)+'\n'
        (BASE/'analysis.json').write_text(raw, encoding='utf8'); destination.write_text(raw, encoding='utf8')
        proof = dict(passed=True, analysis=pin(BASE/'analysis.json'), analyst=pin(Path(__file__)), inputs=value['inputs'],
                     native_sources={p.name: pin(p) for p in (BASE/'native-source').iterdir()}, new_inference_calls=0)
        (BASE/'closed.json').write_text(json.dumps(proof, indent=2)+'\n')
    else: assert read(destination) == value
    print(json.dumps({k: value[k] for k in ['passed', 'new_inference_calls', 'intervals_ms', 'projection_fraction_of_lstm', 'unresolved_fraction_of_lstm']}))
