"""Observe only the closed rational candidate and qualified scalar root."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
OLD=TOOLS.parent/'vector-sigmoid-screen'
QUALIFIED=ROOT/'artifacts/parakeet-pad-current-root-amd-20260926'
BUILD=ROOT/'artifacts/parakeet-rational-sigmoid-build-amd-20260927'
SCREEN=ROOT/'artifacts/parakeet-rational-sigmoid-screen-amd-20260927'


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def read(path):return json.loads(path.read_text(encoding='utf8'))


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def census():
    value=load('original_sigmoid_census',OLD/'census.py').census()
    assert value==read(SCREEN/'bundle/census.json')
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
    expected=[(QUALIFIED,'71c80efd687562cba2e2b5d03e9e036b93de09d0a30f74b976b86355ed12fdf0'),
              (BUILD,'3301ae58b42e1fd51435f54191cf4bba42c6cbf8e2ad6bf4279af67ae5d3d89a'),
              (SCREEN,'fc8a6d9736ff3fd324ebd34ad17a213c857bb178556d4aa7175ee455cc8070b6')]
    evidence={}
    for base,digest in expected:
        assert pin(base/'closed.json')['sha256']==digest
        proof=read(base/'closed.json')
        assert proof['passed'] and proof['analysis']==pin(base/'analysis.json')
        if base==SCREEN:
            assert not proof['admitted'] and not read(base/'analysis.json')['admitted']
            assert pin(OLD/'Screen.cs')==pin(base/'bundle/source/Screen.cs')==proof['files']['bundle/source/Screen.cs']
        for name in ['closed.json','analysis.json']:
            evidence[(base/name).relative_to(ROOT).as_posix()]=pin(base/name)
    review=read(BUILD/'build-review.json')
    assert review['passed'] and read(BUILD/'analysis.json')['compiled_review']==pin(BUILD/'build-review.json')
    assert [r['unchanged'] for r in review['methods']]==[3281,697]
    assert review['data_methods_unchanged'] and review['public_surface_unchanged'] and review['shared_exp_unchanged']
    inventory=BUILD/'build-collected/logs/instructions.json'
    assert pin(inventory)==review['inventory']
    row=read(inventory)['observations'][0]
    product=read(QUALIFIED/'analysis.json')['built']['Lokad.Onnx.dll']
    rejected=review['product']['Lokad.Onnx.dll']
    assert row['before_sha256']==product['sha256'] and row['after_sha256']==rejected['sha256']
    assert len(row['differences'])==len(row['added'])==1 and not row['removed']
    assert '::Sigmoid::' in row['differences'][0] and '::SigmoidRationalVector::' in row['added'][0]
    assert row['method_flags_after'][row['added'][0]]==8
    for path,wanted in [(QUALIFIED/'collected/runtime/Lokad.Onnx.dll',product),
                        (BUILD/'build-collected/runtime/Lokad.Onnx.dll',rejected)]:
        assert pin(path)==wanted;evidence[path.relative_to(ROOT).as_posix()]=wanted
    for path in [inventory,BUILD/'build-review.json',OLD/'Screen.cs']:
        evidence[path.relative_to(ROOT).as_posix()]=pin(path)
    evidence['compiled_scope']=dict(unchanged_core_methods=3281,unchanged_data_methods=697,
        sigmoid_changed=True,one_private_noinline_helper_added=True,diagnostic_only=True)
    return dict(current=product,candidate=rejected),evidence
