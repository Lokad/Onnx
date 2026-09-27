"""Reproduce complete Parakeet correctness without running inference again."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'artifacts/parakeet-decoder-packed-row-models-amd-20260927'
OUT = Path(__file__).with_name('models-20260927.json')


def read(p): return json.loads(p.read_text(encoding='utf8'))


def pin(p):
    with p.open('rb') as f:
        return dict(bytes=p.stat().st_size, sha256=hashlib.file_digest(f, 'sha256').hexdigest())


def summarize():
    closure, analysis = read(BASE/'closed.json'), read(BASE/'analysis.json')
    assert closure['passed'] and analysis['passed'] and analysis['no_performance_measurement']
    assert closure['analysis'] == pin(BASE/'analysis.json')
    for name, wanted in closure['files'].items(): assert pin(BASE/name) == wanted, name
    results = analysis['results']; assert len(results) == 8
    rows, comparisons = [], {}
    arrays = values = requests = 0
    for role in ['selected', 'candidate']:
        for isa in ['512', '256']:
            native = results[f'{role}-native-{isa}']['native']
            public = results[f'{role}-public-{isa}']
            assert native['audit_consistent'] and native['application_passed'] and native['numeric_gate_passed']
            assert not native['failures'] and native['maximum'] <= 1e-4
            assert public['passed'] and public['public_requests'] == 20
            arrays += native['arrays']; values += native['values']; requests += public['public_requests']
            rows.append(dict(role=role, mode='normal' if isa == '512' else 'AVX512 disabled',
                             maximum_scaled_ort_error=native['maximum']))
            if role == 'candidate':
                compared = native['exact_selected_comparisons']
                assert len(compared) == 784 and all(r['bit_identical'] for r in compared)
                assert public['complete_selected_results_exact']
                comparisons[isa] = dict(arrays=len(compared), every_array_byte_identical=True,
                                       complete_public_results_exact=True)
    assert (arrays, values, requests) == (3136, 12361976, 80)
    assert len(analysis['resources']) == 8
    return dict(closure=pin(BASE/'closed.json'), totals=dict(arrays=arrays, values=values, public_requests=requests),
                numeric_rows=rows, current_comparisons=comparisons, release_admitted=False,
                retained_screen_verdict='rejected', **analysis)


if __name__ == '__main__':
    assert sys.argv[1:] in [[], ['--publish']]
    value = summarize()
    if sys.argv[1:]:
        with OUT.open('x', encoding='utf8', newline='\n') as f:
            json.dump(value, f, indent=2, allow_nan=False); f.write('\n')
    else: assert read(OUT) == value
    print(json.dumps({k: value[k] for k in ['passed', 'closure', 'totals', 'numeric_rows', 'current_comparisons']}))
