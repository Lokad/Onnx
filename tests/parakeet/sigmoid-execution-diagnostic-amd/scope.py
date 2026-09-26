"""Bind existing products and preserve every original public call and check."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
OLD = TOOLS.parent / 'vector-sigmoid-screen'
BUILD = ROOT / 'artifacts/parakeet-vector-sigmoid-build-amd-20260925'
QUALIFIED = ROOT / 'artifacts/parakeet-pad-current-root-amd-20260926'
SCREEN = ROOT / 'artifacts/parakeet-vector-sigmoid-screen-amd-20260925'


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def census():
    value = load('original_sigmoid_census', OLD / 'census.py').census()
    assert value == read(SCREEN / 'bundle/census.json')
    return value


EDITS = [
    ('Require(flags.Count==0,"ordinary runtime flags");',
     'Require(CounterProbe.Allowed(flags),"fixed diagnostic logging only");'),
    ('public readonly List<Clock> Clocks = new(780);',
     'public readonly List<Clock> Clocks = new(780);\n        public readonly List<CounterProbe.Observation> Observations = new(780);'),
    ('            long start=Stopwatch.GetTimestamp();',
     '            var observation=CounterProbe.Start();\n            long start=Stopwatch.GetTimestamp();'),
    ('            long ticks=Stopwatch.GetTimestamp()-start;',
     '            long ticks=Stopwatch.GetTimestamp()-start;\n            fixture.Observations.Add(CounterProbe.End(observation,iteration));'),
    ('ownership=true,clocks=fixture.Clocks',
     'ownership=true,clocks=fixture.Clocks,observations=fixture.Observations'),
    ('runtime=Environment.Version.ToString(),pid=Environment.ProcessId,flags,core_sha256=core,',
     'runtime=Environment.Version.ToString(),pid=Environment.ProcessId,flags,core_sha256=core,vector_width=Vector<float>.Count,diagnostic_only=true,'),
]


def instrument(original):
    source = original
    for before, after in EDITS:
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    restored = source
    for before, after in reversed(EDITS):
        assert restored.count(after) == 1, after
        restored = restored.replace(after, before)
    assert restored == original
    assert source.count('CPUExecutionProvider.Sigmoid(fixture.Source,fixture.Options)') == 1
    return source


def references():
    expected = [(QUALIFIED, '71c80efd687562cba2e2b5d03e9e036b93de09d0a30f74b976b86355ed12fdf0'),
                (BUILD, '0b1929e1df62f1dcae578b9c9a6d229b9a074efa19bd9aa0c740b08d5378a07b'),
                (SCREEN, 'aaca2b2fd3b54f66dec841b948c0673279b833f90733770348fcbe41dfc4201f')]
    evidence = {}
    for base, digest in expected:
        assert pin(base / 'closed.json')['sha256'] == digest
        proof = read(base / 'closed.json')
        assert proof['passed'] and proof['analysis'] == pin(base / 'analysis.json')
        if base == SCREEN:
            assert not proof['admitted'] and not read(base / 'analysis.json')['admitted']
            assert pin(OLD / 'Screen.cs') == pin(base / 'bundle/source/Screen.cs') == proof['files']['bundle/source/Screen.cs']
        for name in ['closed.json', 'analysis.json']:
            evidence[(base / name).relative_to(ROOT).as_posix()] = pin(base / name)
    root_path = QUALIFIED / 'collected/inventory/instructions.json'
    old_path = BUILD / 'build-collected/logs/instructions.json'
    assert pin(root_path) == read(QUALIFIED / 'closed.json')['files']['collected/inventory/instructions.json']
    assert pin(old_path) == read(BUILD / 'build-review.json')['inventory']
    before = read(old_path)['observations'][0]
    current = read(root_path)['observations'][0]
    sigmoid, = [k for k in before['normalized_methods'] if '::Sigmoid::' in k]
    assert before['differences'] == [sigmoid] and before['method_flags_before'] == before['method_flags_after']
    checked = []
    for key, body in before['normalized_methods'].items():
        relevant = any(key.startswith('Lokad.Onnx.' + name + '::') for name in
                      ['OpResult', 'ExecutionOptions', 'TensorExecutionOptions', 'Profiler'])
        relevant |= key.startswith(('Lokad.Onnx.DenseTensor`', 'Lokad.Onnx.TensorSlice`'))
        relevant |= key == sigmoid or '::ExpVector::' in key or any('::' + n + '::' in key for n in
                     ['ToDenseTensor', 'get_Dimensions', 'get_IsReversedStride', 'get_Length', 'get_ElementType'])
        if relevant:
            assert current['normalized_methods'][key] == body, key
            assert current['method_flags_after'][key] == before['method_flags_before'][key], key
            checked.append(key)
    assert sigmoid in checked and any('::ExpVector::' in k for k in checked)
    product = read(QUALIFIED / 'analysis.json')['built']['Lokad.Onnx.dll']
    rejected = read(BUILD / 'build-review.json')['product']['Lokad.Onnx.dll']
    assert current['after_sha256'] == product['sha256'] and before['after_sha256'] == rejected['sha256']
    for path, wanted in [(QUALIFIED / 'collected/runtime/Lokad.Onnx.dll', product),
                         (BUILD / 'build-collected/runtime/Lokad.Onnx.dll', rejected)]:
        assert pin(path) == wanted
        evidence[path.relative_to(ROOT).as_posix()] = wanted
    for path in [root_path, old_path, BUILD / 'build-review.json', OLD / 'Screen.cs']:
        evidence[path.relative_to(ROOT).as_posix()] = pin(path)
    evidence['relevant_method_scope'] = dict(methods=len(checked), sigmoid_equal=True,
        exp_vector_equal=True, allocation_and_options_helpers_equal=True,
        historical_baseline_core=before['before_sha256'], current_core=product['sha256'],
        diagnostic_only=True, unrelated_product_methods_not_claimed_equal=True)
    return dict(current=product, candidate=rejected), evidence
