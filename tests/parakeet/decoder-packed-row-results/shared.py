"""Reproduce the completed shared/e5 verdict from retained files, without inference."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'artifacts/parakeet-decoder-packed-row-shared-amd-20260927'
OUT = Path(__file__).with_name('shared-20260927.json')


def read(path): return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def summarize():
    assert pin(BASE/'closed.json')['sha256'] == 'bdd1c7664d902af7d135318f12856419a61d222e69bb073ae24e10b8aca8aca0'
    proof, analysis = read(BASE/'closed.json'), read(BASE/'analysis.json')
    assert proof['passed'] and analysis['passed'] and analysis['no_performance_measurement']
    assert proof['analysis'] == pin(BASE/'analysis.json') and analysis['reference_provenance_verified']
    for name, wanted in proof['files'].items(): assert pin(BASE/name) == wanted, name
    results = analysis['results']
    assert set(results) == {'selected-shared', 'selected-e5', 'candidate-shared', 'candidate-e5'}
    rows = []
    for role in ['selected', 'candidate']:
        assert sum(results[role+'-'+mode]['arrays'] for mode in ['shared', 'e5']) == 166
        assert sum(results[role+'-'+mode]['values'] for mode in ['shared', 'e5']) == 5000814
        for mode in ['shared', 'e5']:
            result = results[role+'-'+mode]
            assert result['passed'] and result['no_performance_measurement']
            maximum = max(r['maximum'] for r in result['rows'])
            assert maximum <= 1e-4
            if role == 'candidate': assert all(r['exact_selected'] for r in result['rows'])
            rows.append(dict(role=role, mode=mode, arrays=result['arrays'], values=result['values'],
                             maximum_scaled_ort_error=maximum, exact_current=True if role == 'candidate' else None))
    return dict(passed=True, closure=pin(BASE/'closed.json'), identities=analysis['identities'],
                consumer=analysis['consumer'], rows=rows, totals=dict(arrays=332, values=10001628),
                resources=analysis['resources'], reference_provenance_verified=True,
                no_performance_measurement=True, release_admitted=False,
                retained_component_screen_verdict='rejected')


if __name__ == '__main__':
    assert sys.argv[1:] in [[], ['--publish']]
    value = summarize()
    if sys.argv[1:]:
        with OUT.open('x', encoding='utf8', newline='\n') as stream:
            json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
    else: assert read(OUT) == value
    print(json.dumps({k: value[k] for k in ['passed', 'closure', 'totals', 'rows']}))
