"""Reproduce all individual-call observations and the post-capture cache receipt."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'artifacts/parakeet-decoder-unmapped-calls-amd-20260927'
TOOLS = ROOT/'tests/parakeet/decoder-unmapped-calls'


def pin(path):
    with path.open('rb') as f: return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(f, 'sha256').hexdigest())


def read(path): return json.loads(path.read_text())


def module(name, path):
    loader = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(loader); loader.loader.exec_module(value)
    return value


def summarize():
    assert pin(BASE/'closed.json')['sha256'] == '966941d1e040211b840259438fe8bb786aea23dc3c8f9987208cf242b5a7e986'
    closed = read(BASE/'closed.json'); assert closed['passed'] and not closed['admitted']
    for name, wanted in closed['files'].items(): assert pin(BASE/name) == wanted, name
    prepared = read(BASE/'prepared.json'); path = TOOLS/'observations.py'
    assert pin(path) == prepared['tools']['observations.py']
    observation = module('unmapped_observation', path)
    path = ROOT/'tests/parakeet/decoder-packed-row-screen/score.py'
    assert pin(path) == prepared['helpers'][path.relative_to(ROOT).as_posix()]
    scoring = module('unmapped_original_score', path)
    folder = BASE/'capture-collected'
    reports = {n: read(folder/'logs'/(n+'.json')) for n in observation.ORDER}
    result = observation.analyze(reports); analysis = read(BASE/'analysis.json')
    assert all(analysis[k] == v for k, v in result.items())
    comparison = scoring.score(reports, read(folder/'census.json'))
    assert comparison == analysis['instrumented_comparison']
    topology_path = ROOT/'artifacts/parakeet-decoder-unmapped-cache-topology-20260927.json'
    assert pin(topology_path)['sha256'] == '73bc117defa4ce464f9d2e129ee5d3e004576b8fac0bed9d35fc7e44beac13df'
    topology = read(topology_path)
    assert topology['passed'] and topology['boot'] == 1789634288.0
    assert topology['terminal_owners'] == closed['terminal_owners'] and topology['cpu'] == 2
    assert topology['cache']['index3']['size'] == '32768K'
    case = read(folder/'census.json')['cases'][0]
    weight_bytes = 4*case['reduction']*case['shape'][-1]
    return dict(passed=True, **result, closure=pin(BASE/'closed.json'),
        products=analysis['products'], consumer=analysis['consumer'],
        calls=comparison['calls'], exact_output_checks=comparison['exact_output_checks'],
        instrumented_rows=comparison['rows'], instrumented_controls=comparison['controls'],
        original_screen_admitted=False, output_hashes_equal=True,
        reported_cache=topology['cache'], cache_receipt=pin(topology_path),
        one_weight_bytes=weight_bytes, two_representations_bytes=2*weight_bytes,
        reported_l3_bytes=32768*1024, resources=analysis['resources'],
        maximum_foreign_cpu_fraction=max(v['foreign_cpu_fraction'] for v in analysis['foreign_cpu']))


if __name__ == '__main__':
    result = summarize(); target = Path(__file__).with_name('unmapped-calls-20260927.json')
    if '--publish' in sys.argv:
        with target.open('x') as f: json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    else: assert read(target) == result
    print(json.dumps(dict(passed=True, report=pin(target), first_call_explanation_supported=result['first_call_explanation_supported'],
        batch_ratio=result['batch_ratio']['value'], position_ratios=[r['ratio']['value'] for r in result['positions']],
        release_admitted=False)))
