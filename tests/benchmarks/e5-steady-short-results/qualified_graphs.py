"""Combine complete case verdicts while preserving the original failed campaign."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GRAPH = ROOT/'artifacts/parakeet-owned-batch-isolation-graphs-amd-20260925'
E5 = ROOT/'artifacts/e5-steady-short-amd-20260925'
BASE = ROOT/'artifacts/parakeet-owned-batch-graph-qualification-20260925'
ORDER = ['current-a', 'candidate-a', 'ort-a', 'ort-b', 'candidate-b', 'current-b']


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def verified(folder):
    proof = read(folder/'closed.json')
    assert proof['passed']
    for name, wanted in proof['files'].items():
        assert pin(folder/name) == wanted, name
    return proof, read(folder/'analysis.json')


def scorer(path, name, preparation):
    assert pin(path) == read(preparation/'prepared.json')['files'][path.relative_to(ROOT).as_posix()]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def derive():
    original, old = verified(GRAPH)
    successor, new = verified(E5)
    assert pin(GRAPH/'closed.json')['sha256'] == 'def19d3f178cbb318bc11999b6e23dd18db40772d78949707155cf1f4c791638'
    assert not original['admitted'] and not original['all_controls_passed']
    assert [r['key'] for r in old['performance'] if not r['qualified']] == ['e5-8tok']
    assert old['clocks'] == 41112 and old['measured'] == 8640
    assert new['clocks'] == 37089 and new['measured'] == 1080 and len(new['setups']) == 9
    assert new['original_graph_closure'] == pin(GRAPH/'closed.json')
    assert new['diagnosis_closure']['sha256'] == '5b87937aa914ab9007928bfb3ba0f90c99eae3ff8f3d23611467cbb98ab8ed4b'
    assert not old['root_product_changed'] and not new['root_product_changed']
    for analysis in [old, new]:
        assert analysis['consumer']['implementation_flags_equal'] and analysis['consumer']['branches_locals_exceptions_equal']
    assert [(r['before'], r['after']) for r in new['consumer']['changes']] == [(780,6180), (600,6000)]
    products = read(GRAPH/'payload.json')['products']
    assert products == read(E5/'payload.json')['products'] == new['products']
    parent = ROOT/'tests/parakeet/packed-final-row-graphs-amd'
    baseline = scorer(parent/'statistics_base.py', 'qualified_original_base', GRAPH)
    long_e5 = scorer(parent/'statistics_e5.py', 'qualified_original_e5', GRAPH)
    short_e5 = scorer(ROOT/'tests/benchmarks/e5-steady-short-amd/statistics.py', 'qualified_short_e5', E5)
    performance, clocks, setups = [], [], []
    for row in old['performance']:
        key = row['key']
        use_new = key == 'e5-8tok'
        folder = E5 if use_new else GRAPH
        source = 'short-e5-correction' if use_new else 'original-graphs'
        stats = short_e5 if use_new else long_e5 if key == 'e5-30tok' else baseline
        values = {role: read(folder/'collected'/f'timing-{key}-{role}'/'output/result.json') for role in ORDER}
        result = stats.summarize(values)
        expected = new['performance'] if use_new else row
        assert expected == dict(key=key, **result)
        performance.append(dict(key=key, source=source,
            warmups=6000 if use_new else 1200 if key == 'e5-30tok' else 600,
            measured_per_process=180, **result))
        names = [f'verify-{key}-{role}' for role in ['current', 'candidate', 'ort']]
        names += [f'timing-{key}-{role}' for role in ORDER]
        for name in names:
            value = read(folder/'collected'/name/'output/result.json')
            assert value['passed'] and value['inputs_unchanged'] and value['held_outputs_unchanged'] and value['flags'] == {}
            role = name.split('-')[-1] if name.startswith('verify-') else name.split('-')[-2]
            if role != 'ort':
                assert value['core'] == products[role]['Lokad.Onnx.dll']['sha256']
            assert all(0 <= a['max_scaled_error'] <= 1e-4 for a in value['arrays'])
            clocks.extend(dict(source=source, process=name, **clock) for clock in value['clocks'])
            setups.append(dict(source=source, process=name, seconds=value['setup_seconds']))
    assert len(performance) == 8 and len(clocks) == 73512
    assert sum(not c['warmup'] for c in clocks) == 8640 and len(setups) == 72
    assert successor['admitted'] == new['performance']['qualified']
    result = dict(passed=True, admitted=all(r['qualified'] for r in performance),
        all_controls_passed=all(c['passed'] for r in performance for c in r['controls']),
        products=products, performance=performance, clocks=len(clocks), measured=8640, setups=setups,
        source_closures=dict(original_graphs=pin(GRAPH/'closed.json'), short_e5_correction=pin(E5/'closed.json')),
        source_calls_retained=78201, original_graph_failure_preserved=True, root_product_changed=False,
        consumer=dict(original=old['consumer'], original_e5=old['e5_consumer'], short_e5=new['consumer']))
    return result, clocks


def qualify():
    assert not BASE.exists()
    analysis, clocks = derive()
    BASE.mkdir()
    (BASE/'analysis.json').write_text(json.dumps(analysis, indent=2, allow_nan=False)+'\n', encoding='utf8')
    with (BASE/'clocks.csv').open('x', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(clocks[0]))
        writer.writeheader()
        writer.writerows(clocks)
    closure = dict(passed=True, admitted=analysis['admitted'], all_controls_passed=analysis['all_controls_passed'],
        source_closures=analysis['source_closures'], generator=pin(Path(__file__)),
        files={p.name: pin(p) for p in BASE.iterdir() if p.is_file()})
    (BASE/'closed.json').write_text(json.dumps(closure, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(closed=pin(BASE/'closed.json'), admitted=analysis['admitted'], clocks=analysis['clocks'])))


def admission():
    proof, value = verified(BASE)
    actual, _ = derive()
    assert value == actual and proof['source_closures'] == value['source_closures']
    assert proof['admitted'] == value['admitted'] and proof['all_controls_passed'] == value['all_controls_passed']
    assert proof['generator'] == pin(Path(__file__))
    assert proof['admitted'] and proof['all_controls_passed']
    return value


if __name__ == '__main__':
    qualify()
