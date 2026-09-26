"""Publish the closed current-root Pad correctness result without rescoring."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-pad-current-models-amd-20260926'
SCREEN = ROOT/'artifacts/parakeet-pad-current-screen-amd-20260926'


def read(path): return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def main():
    assert not (OUT/'models-20260926.json').exists()
    closure, value = read(BASE/'closed.json'), read(BASE/'analysis.json')
    assert pin(BASE/'closed.json')['sha256'] == '194eb9a48b64df22d19b51adfc3ebe3accc56004544620123c128c2a76f066bf'
    assert closure['passed'] and value['passed'] and closure['analysis'] == pin(BASE/'analysis.json')
    for name,wanted in closure['files'].items(): assert pin(BASE/name) == wanted, name
    assert value['no_performance_measurement'] and value['reference_provenance_verified']
    assert len(value['resources']) == len(value['results']) == 8
    screen = read(SCREEN/'analysis.json')
    assert not read(SCREEN/'closed.json')['admitted'] and not screen['admitted']
    failures = [r for r in screen['controls'] if not r['passed']]
    assert len(failures) == 6
    assert value['identities'] == dict(selected=screen['products']['current'],candidate=screen['products']['candidate'])
    rows = []
    for name,result in value['results'].items():
        assert result['passed']
        if '-native-' in name:
            native = result['native']
            assert native['numeric_gate_passed'] and native['audit_consistent'] and native['application_passed'] and not native['failures']
            assert (native['arrays'],native['values']) == (784,3090494)
            exact = native.get('exact_selected_comparisons',[])
            if name.startswith('candidate-'): assert len(exact) == 784 and all(r['bit_identical'] for r in exact)
            rows.append(dict(name=name,arrays=784,values=3090494,maximum=native['maximum'],exact_selected_arrays=len(exact)))
        else:
            assert result['public_requests'] == 20
            if name.startswith('candidate-'): assert result['complete_selected_results_exact']
            rows.append(dict(name=name,public_requests=20))
    assert sum(r.get('arrays',0) for r in rows) == 3136
    assert sum(r.get('values',0) for r in rows) == 12361976
    assert sum(r.get('public_requests',0) for r in rows) == 80
    maximum = max(r['maximum'] for r in rows if 'maximum' in r)
    samples = sum(r['samples'] for r in value['resources'])
    peak = max(r['peak_rss'] for r in value['resources'])
    published = dict(closure=pin(BASE/'closed.json'),analysis=pin(BASE/'analysis.json'),
        identities=value['identities'],consumers=value['consumers'],jobs=rows,resources=value['resources'],
        terminal_owners=closure['remote_terminal'],component_screen=pin(SCREEN/'closed.json'),
        failed_component_controls=failures,no_performance_measurement=True,release_admitted=False,
        publisher=pin(Path(__file__)))
    report = f'''# Current padding dispatcher: full Parakeet correctness passes

All eight workers pass. Across both products and normal/AVX512-disabled execution,
**3,136 arrays / 12,361,976 values** satisfy the pinned Microsoft ORT reference,
and **80 complete transcriptions** pass over the 20 clips. Candidate tensors and
public results match the qualified current root exactly, including transcripts,
tokens and decoder decisions. Input immutability and independently held outputs
pass. Maximum scaled ORT error is **{maximum:.9g}**, below the unchanged **1e-4**.

Current Core f3992f40 / Data a8e0b583 and candidate Core a74acb17 / Data be954dc4
use the same existing compiled consumers. No product, consumer or model was
rebuilt. Every owner is terminal/code zero; all {samples:,} resource observations
pass, with peak owned RSS {peak:,} bytes. These elapsed times are correctness
execution costs, not performance scores.

The component screen remains failed with all six repeatability failures retained.
The [memory diagnosis](../pad-memory-results/managed-diagnosis-20260926.md) and
explicit plan decision permit this complete-model test; neither changes the
component verdict. The next decision is one complete transcription comparison
against the actual current root and ORT, with the original 3% improvement,
per-clip and repeatability gates. Shared/Pyannote, graph and root/package
qualification remain required before any source or BENCHMARK.md promotion.

[Exact identities, totals and failures](models-20260926.json),
[correctness protocol](../pad-current-models-amd/README.md),
[application protocol](../pad-current-app-amd/README.md).

Closure: `{pin(BASE/'closed.json')['sha256']}`.
'''
    with (OUT/'models-20260926.json').open('x',encoding='utf8') as stream:
        json.dump(published,stream,indent=2,allow_nan=False);stream.write('\n')
    with (OUT/'models-20260926.md').open('x',encoding='utf8') as stream: stream.write(report)
    print(json.dumps(dict(passed=True,arrays=3136,values=12361976,public_requests=80,maximum=maximum,resources=samples)))


if __name__ == '__main__': main()
