"""Reproduce the rejected fixed comparison without executing a model."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'artifacts/parakeet-decoder-packed-row-screen-v2-amd-20260927'
TOOLS = ROOT/'tests/parakeet/decoder-packed-row-screen'
import hashlib


def pin(path):
    with path.open('rb') as f: return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(f, 'sha256').hexdigest())


def read(path): return json.loads(path.read_text())


def summarize():
    assert pin(BASE/'closed.json')['sha256'] == '3e6a7562db1946954f2cddba10b2ee31958854e66c50fbb55848ed04e3e75148'
    closure = read(BASE/'closed.json'); assert closure['passed'] and not closure['admitted']
    for name, wanted in closure['files'].items(): assert pin(BASE/name) == wanted, name
    prepared = read(BASE/'prepared.json')
    path = TOOLS/'score.py'; assert pin(path) == prepared['tools']['score.py']
    loader = importlib.util.spec_from_file_location('packed_row_clock_score', path)
    module = importlib.util.module_from_spec(loader); loader.loader.exec_module(module)
    folder = BASE/'capture-collected'; reports = {n: read(folder/'logs'/(n+'.json')) for n in module.ORDER}
    score = module.score(reports, read(folder/'census.json')); analysis = read(BASE/'analysis.json')
    assert all(analysis[k] == v for k, v in score.items())
    blocks = {}
    for name, report in reports.items():
        blocks[name] = {}
        for row in report['rows']:
            blocks[name][row['name']] = [sum(c['ticks'] for c in row['clocks'][i:i+60])/60/report['frequency']/row['batch']
                for i in range(0, 780, 60)]
    return dict(passed=True, admitted=False, release_admitted=False, closure=pin(BASE/'closed.json'),
        products=analysis['products'], consumer=analysis['consumer'],
        **{k: score[k] for k in ['rows', 'controls', 'gates', 'processes', 'accounting', 'calls', 'samples',
            'warmup_calls', 'measured_calls', 'exact_output_checks']},
        all_input_weight_output_hashes_equal=True, resources=analysis['resources'],
        maximum_foreign_cpu_fraction=max(r['foreign_cpu_fraction'] for r in analysis['foreign_cpu']),
        sixty_round_blocks_seconds=blocks,
        limitation='Blocks describe every original warmup and measured clock; they never replace the full-mean failed verdict.')


if __name__ == '__main__':
    result = summarize(); target = Path(__file__).with_name('screen-20260927.json')
    if '--publish' in sys.argv:
        with target.open('x') as f: json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    else: assert read(target) == result
    print(json.dumps(dict(passed=True, admitted=False, report=pin(target), calls=result['calls'],
        rows=[dict(name=r['name'], ratio=r['ratio']['value'], passed=r['passed']) for r in result['rows']])))
