"""Reproduce completed Pyannote correctness from retained evidence only."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'artifacts/parakeet-decoder-lstm-layout-pyannote-amd-20260927'
OUT = Path(__file__).with_name('pyannote-20260927.json')


def read(path): return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def summarize():
    assert pin(BASE/'closed.json')['sha256'] == '8402f31da2de3034f8f64a6ed30644b4fb0e527c72df77b2c2c04113c747e808'
    proof, analysis = read(BASE/'closed.json'), read(BASE/'analysis.json')
    assert proof['passed'] and analysis['passed'] and proof['analysis'] == pin(BASE/'analysis.json')
    for name, wanted in proof['files'].items(): assert pin(BASE/name) == wanted, name
    assert analysis['no_performance_measurement'] and analysis['reference_provenance_verified']
    assert not analysis['consumer_rebuilt'] and not analysis['consumer_reuse']['product_rebuilt']
    guards = analysis['identity_guards']
    assert guards['passed'] and guards['probes'] == 4 and guards['rejection_before_output']
    assert guards['consumer']['sha256'] == 'd78c45b9059c0f22e55c5112ad12fe54b34d1e0567096f52d23b67d3b141a848'
    rows = []
    for role in ['selected', 'candidate']:
        result = analysis['results'][role]
        assert result['passed'] and (result['arrays'], result['values'], result['public_calls']) == (18, 2917107, 16)
        native = [c for c in result['comparisons'] if c['reference'] == 'native']
        assert len(native) == 18 and all(c['failed_values'] == 0 for c in result['comparisons'])
        if role == 'candidate':
            compared = [c for c in result['comparisons'] if c['reference'] == 'production']
            assert len(compared) == 18 and all(c['bit_identical'] for c in compared)
            assert result['complete_public_results_exact'] and result['complete_public_semantics_exact']
        maximum = max(c['maximum'] for c in native); assert maximum <= 1e-4
        rows.append(dict(role=role, maximum_scaled_ort_error=maximum))
    return dict(closure=pin(BASE/'closed.json'), totals=dict(arrays=36, values=5834214, public_requests=32),
                numeric_rows=rows, release_admitted=False, retained_component_screen_verdict='rejected', **analysis)


if __name__ == '__main__':
    assert sys.argv[1:] in [[], ['--publish']]
    value = summarize()
    if sys.argv[1:]:
        with OUT.open('x', encoding='utf8', newline='\n') as stream:
            json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
    else: assert read(OUT) == value
    print(json.dumps({k: value[k] for k in ['passed', 'closure', 'totals', 'numeric_rows']}))
