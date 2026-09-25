"""Publish complete Parakeet correctness for the graph-qualified relocation."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-owned-batch-isolation-models-amd-20260925'
GRAPHS = ROOT/'artifacts/parakeet-owned-batch-graph-qualification-20260925'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def main():
    assert not (OUT/'models-20260925.json').exists()
    closure, value = read(BASE/'closed.json'), read(BASE/'analysis.json')
    assert closure['passed'] and value['passed'] and closure['analysis'] == pin(BASE/'analysis.json')
    for name, wanted in closure['files'].items():
        assert pin(BASE/name) == wanted, name
    assert value['no_performance_measurement'] and value['reference_provenance_verified']
    assert len(value['resources']) == len(value['results']) == 8
    assert pin(GRAPHS/'closed.json')['sha256'] == 'ec95b9c7f8019b402fe513b6da6938cf83a8bc2c91f1d5ab65fedd4f7b55ed7c'
    assert read(GRAPHS/'closed.json')['admitted']
    identities = value['identities']
    assert identities['selected']['Lokad.Onnx.dll']['sha256'].startswith('40260aef')
    assert identities['candidate']['Lokad.Onnx.dll']['sha256'] == 'e07a45189b348fe55ce76300415c6c35ba6a2bc0d224f1fc13b0b92c303bccba'
    assert identities['selected']['Lokad.Onnx.Data.dll'] == identities['candidate']['Lokad.Onnx.Data.dll']
    rows = []
    for name, result in value['results'].items():
        assert result['passed']
        if '-native-' in name:
            native = result['native']
            assert native['numeric_gate_passed'] and native['audit_consistent'] and native['application_passed'] and not native['failures']
            assert (native['arrays'], native['values']) == (784, 3090494)
            exact = native.get('exact_selected_comparisons', [])
            if name.startswith('candidate-'):
                assert len(exact) == 784 and all(r['bit_identical'] for r in exact)
            rows.append(dict(name=name, arrays=native['arrays'], values=native['values'],
                maximum=native['maximum'], exact_parent_arrays=len(exact), passed=True))
        else:
            assert result['public_requests'] == 20
            if name.startswith('candidate-'):
                assert result['complete_selected_results_exact']
            rows.append(dict(name=name, public_requests=20, passed=True))
    assert sum(r.get('arrays', 0) for r in rows) == 3136
    assert sum(r.get('values', 0) for r in rows) == 12361976
    assert sum(r.get('public_requests', 0) for r in rows) == 80
    result = dict(closure=pin(BASE/'closed.json'), analysis=pin(BASE/'analysis.json'),
        graph_qualification=pin(GRAPHS/'closed.json'), identities=identities, consumers=value['consumers'],
        jobs=rows, resources=value['resources'], terminal_owners=closure['remote_terminal'],
        no_performance_measurement=True, release_admitted=False, publisher=pin(Path(__file__)))
    maximum = max(r['maximum'] for r in rows if 'maximum' in r)
    peak = max(r['peak_rss'] for r in value['resources'])
    samples = sum(r['samples'] for r in value['resources'])
    report = f'''# Dispatch relocation: complete Parakeet correctness passes

All eight workers pass. Across both products and normal/AVX512-disabled modes,
the audit checks **3,136 arrays / 12,361,976 values** against pinned Microsoft
ORT truth, plus **80 complete public transcription requests** over the 20 clips.
Candidate tensors and complete public results match the direct-depthwise parent
exactly, including transcripts, tokens and decoder decisions. Input immutability
and independently retained outputs pass. Maximum scaled ORT error is
**{maximum:.9g}**, below the unchanged **1e-4** bound.

The same compiled consumers exercise normal ParakeetTranscriber construction
and actual encoder/decoder execution. Parent Core40260aef and candidate Coree07a4518
both use unchanged Data01e9e784. Every recorded owner is terminal/code0. All
{samples:,} resource observations pass; peak owned RSS is {peak:,} bytes.
These are correctness checks; their elapsed times are not performance scores.

The [combined graph qualification](../../benchmarks/e5-steady-short-results/qualified-graphs-20260925.md)
already passes all eight cases. Next are fresh full-request comparisons against
the successful depthwise parent and actual released product, followed by the
remaining shared/Pyannote and root/package requirements. No source or BENCHMARK.md
promotion follows from correctness alone.

[Exact identities, job totals and resources](models-20260925.json),
[full-model protocol](../owned-batch-isolation-models/README.md).

Closure: `{pin(BASE/'closed.json')['sha256']}`.
'''
    with (OUT/'models-20260925.json').open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    with (OUT/'models-20260925.md').open('x', encoding='utf8') as stream:
        stream.write(report)
    print(json.dumps(dict(passed=True, arrays=3136, values=12361976, public_requests=80,
        maximum=maximum, resources=samples, closure=pin(BASE/'closed.json'))))


if __name__ == '__main__':
    main()
