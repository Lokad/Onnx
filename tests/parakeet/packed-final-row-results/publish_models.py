"""Publish complete native/public correctness without treating it as a speed score."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-packed-final-row-models-amd-20260925'
CENSUS=ROOT/'artifacts/parakeet-packed-final-row-census-resume-amd-20260925'


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def read(path):return json.loads(path.read_text(encoding='utf8'))


def main():
    assert pin(BASE/'payload.json')['sha256']=='df9b99885df5a1a982ef3468b45b96c7cddd17d94dfcc6538f9d9357c89026e4'
    closure=read(BASE/'closed.json');value=read(BASE/'analysis.json')
    assert closure['passed'] and value['passed'] and closure['analysis']==pin(BASE/'analysis.json')
    for name,wanted in closure['files'].items():assert pin(BASE/name)==wanted,name
    assert value['no_performance_measurement'] and value['reference_provenance_verified']
    assert len(value['resources'])==len(value['results'])==8
    assert pin(CENSUS/'closed.json')['sha256']=='fa159300cbe217b8ac08df793031850f2794f2149dc65b89e89b93a5d5692a77'
    assert value['identities']['candidate']==read(CENSUS/'analysis.json')['product']
    rows=[]
    for name,result in value['results'].items():
        assert result['passed']
        if '-native-' in name:
            native=result['native']
            assert native['numeric_gate_passed'] and native['audit_consistent'] and native['application_passed'] and not native['failures']
            assert (native['arrays'],native['values'])==(784,3090494)
            exact=native.get('exact_selected_comparisons',[])
            if name.startswith('candidate-'):assert len(exact)==784 and all(r['bit_identical'] for r in exact)
            rows.append(dict(name=name,arrays=native['arrays'],values=native['values'],maximum=native['maximum'],
                exact_selected_arrays=len(exact),passed=True))
        else:
            assert result['public_requests']==20
            if name.startswith('candidate-'):assert result['complete_selected_results_exact']
            rows.append(dict(name=name,public_requests=20,passed=True))
    assert sum(r.get('arrays',0) for r in rows)==3136 and sum(r.get('values',0) for r in rows)==12361976
    assert sum(r.get('public_requests',0) for r in rows)==80
    paths=[OUT/('models-20260925'+suffix) for suffix in ['.json','.md']]
    assert not any(p.exists() for p in paths)
    maintenance=ROOT/'artifacts/parakeet-m78-model-headroom-recovery-20260925/closed.json'
    retired=read(maintenance);assert retired['passed'] and retired['no_model_worker_during_retirement'] and retired['resumed_same_owner']
    result=dict(closure=pin(BASE/'closed.json'),analysis=pin(BASE/'analysis.json'),census=pin(CENSUS/'closed.json'),
        identities=value['identities'],consumers=value['consumers'],jobs=rows,resources=value['resources'],
        terminal_owners=closure['remote_terminal'],maintenance=pin(maintenance),no_performance_measurement=True,
        release_admitted=False,failed_release_controls=read(CENSUS/'analysis.json')['failed_release_controls'],publisher=pin(Path(__file__)))
    maximum=max(r['maximum'] for r in rows if 'maximum' in r)
    peak=max(r['peak_rss'] for r in value['resources'])
    report=f'''# Parakeet: complete correctness for the packed final-row fix

**All eight workers pass.** Across both products and both instruction modes,
the audit checks 3,136 arrays containing 12,361,976 values against pinned ORT
truth, plus 80 complete public transcription requests. Every candidate array
is bit-identical to the selected M73 baseline. All public transcripts, tokens,
frame decisions and completion results match; input immutability and independent
held outputs remain verified. Maximum scaled ORT error is {maximum:.9g}, within
the unchanged 1e-4 bound.

The same compiled consumers and numerical/public auditors exercise normal
ParakeetTranscriber construction, the actual encoder and all 20 clips. Both
normal and AVX512-disabled execution pass. Candidate Core49901366/Data01e9e784
are unchanged from the compiled proof and focused contracts. Peak monitored
process RSS is {peak:,} bytes. These runs provide correctness evidence and do
not score application latency.

The same supervisor completes all eight jobs once. After five jobs, it waits
below the unchanged free-memory threshold. Scoped maintenance pauses only that
supervisor, proves no model worker is running, retires 3,136 older M76 VM tensor
duplicates while preserving every exact local original, and resumes the same
owner. The initial pause-acknowledgement race is retained separately; no tensor
was deleted by that failed attempt and no model job was repeated. All model
inputs, limits, assertions and raw outputs remain unchanged.

Next, confirm zero remaining-row reconstruction on the actual corpus, preserving
the measured packing reduction, then run the fresh six-process application
comparison with ORT. The original gain/repeatability gates and unresolved e5
release controls remain. BENCHMARK.md stays on the qualified repository product.

[Exact identities, job totals and resource evidence](models-20260925.json),
[real-model weight census](census-20260925.md),
[full-corpus protocol](../packed-final-row-models-amd/README.md).

Closure: `{pin(BASE/'closed.json')['sha256']}`.
'''
    with paths[0].open('x',encoding='utf8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    with paths[1].open('x',encoding='utf8') as stream:stream.write(report)
    print(json.dumps(dict(passed=True,arrays=3136,values=12361976,public_requests=80,maximum=maximum,reports={p.name:pin(p) for p in paths})))


if __name__=='__main__':main()
