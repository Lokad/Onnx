"""Publish complete-model correctness, retaining the failed operator verdict."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-rational-sigmoid-models-amd-20260927'


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def read(path): return json.loads(path.read_text(encoding='utf8'))


def main():
    closure, analysis = read(BASE/'closed.json'), read(BASE/'analysis.json')
    assert closure['passed'] and analysis['passed'] and analysis['no_performance_measurement']
    assert closure['analysis'] == pin(BASE/'analysis.json')
    for name,wanted in closure['files'].items(): assert pin(BASE/name) == wanted, name
    results = analysis['results']; assert len(results) == 8
    rows, cross = [], {}
    arrays = values = requests = 0
    for role in ['selected','candidate']:
        for isa in ['512','256']:
            native = results[f'{role}-native-{isa}']['native']
            public = results[f'{role}-public-{isa}']
            assert native['audit_consistent'] and native['application_passed'] and native['numeric_gate_passed']
            assert not native['failures'] and native['maximum'] <= 1e-4
            assert public['passed'] and public['public_requests'] == 20
            arrays += native['arrays']; values += native['values']; requests += public['public_requests']
            rows.append(dict(role=role,mode='normal' if isa=='512' else 'AVX512 disabled',
                             maximum_scaled_ort_error=native['maximum']))
            if role == 'candidate':
                assert public['complete_selected_results_exact']
                compared = native['selected_comparisons']; assert len(compared) == 784
                worst = max(compared,key=lambda r:r['maximum_scaled_error'])
                cross[isa] = dict(arrays=len(compared),bit_identical_arrays=sum(r['bit_identical'] for r in compared),
                    maximum_scaled_current_error=worst['maximum_scaled_error'],worst=worst,
                    exact_integer_arrays=sum(r['dtype']!='Float' and r['bit_identical'] for r in compared))
    assert (arrays,values,requests) == (3136,12361976,80)
    resources = analysis['resources']; assert len(resources) == 8
    samples = sum(r['samples'] for r in resources); peak = max(r['peak_rss'] for r in resources)
    paths = [TOOLS/'models-20260927.md',TOOLS/'models-20260927.json']
    assert not any(p.exists() for p in paths)
    payload = dict(closure=pin(BASE/'closed.json'),totals=dict(arrays=arrays,values=values,public_requests=requests),
                   numeric_rows=rows,current_comparisons=cross,**analysis)
    lines = ['# Rational sigmoid: complete Parakeet correctness passes','',
        '**All eight workers pass.** Across current and candidate products in normal and',
        'AVX512-disabled execution, 3,136 arrays / 12,361,976 values satisfy the retained',
        'Microsoft ORT reference. All 80 complete public transcriptions pass over the',
        'twenty clips. Integers, decoder decisions, tokens and public results remain',
        'exact; immutable inputs and independently held outputs pass. The unchanged',
        'scaled-error bound is `abs(actual-reference) / max(1,abs(reference)) <= 1e-4`.','',
        '| Product | Execution | Maximum scaled ORT error |',
        '| --- | --- | ---: |']
    lines += [f"| {r['role']} | {r['mode']} | {r['maximum_scaled_ort_error']:.10g} |" for r in rows]
    lines += ['', 'The arithmetic change can alter floating-point bits. All 784 candidate arrays',
        'per execution mode are also compared directly to current with the same scaled',
        'bound; integer outputs must remain byte-identical. Every difference is recorded.','',
        '| Execution | Maximum scaled error versus current | Byte-identical arrays |',
        '| --- | ---: | ---: |']
    lines += [f"| {'Normal' if isa=='512' else 'AVX512 disabled'} | {r['maximum_scaled_current_error']:.10g} | {r['bit_identical_arrays']} / 784 |" for isa,r in cross.items()]
    lines += ['', 'Current Core `8bb22038` / Data `d02dbf55` and candidate Core `946ddfb6` /',
        'Data `dbe95936` use the original compiled consumers and model fixtures. No',
        'consumer or model was rebuilt or downloaded for this campaign. All owners are',
        f'terminal with code zero; all {samples:,} resource observations pass, with peak',
        f'owned RSS {peak:,} bytes. Job elapsed times are correctness execution costs,',
        '**not application performance measurements**.','',
        'The operator screen remains rejected: 13 failed repeatability controls, four',
        'failed fallback regressions and a missed 75% weighted-gain threshold. The',
        'subsequent diagnostic leaves double latency unresolved. This numerical result',
        'does not change either conclusion and does not admit the candidate for release.','',
        'The next independent decision is one matched complete-transcription comparison',
        'with the original >=3% corpus gain, <=5% per-clip regression and repeatability',
        'gates. Shared/Pyannote, graph and root/package qualification plus an explicit',
        'assessment of retained fallback risk remain necessary before source promotion.',
        'The current release and BENCHMARK.md remain unchanged.','',
        '[All numerical differences, identities and resources](models-20260927.json),',
        '[model protocol](../rational-sigmoid-models-amd/README.md),',
        '[failed screen](screen-20260927.md),',
        '[fallback diagnosis](fallback-diagnosis-20260927.md),',
        '[prospective application protocol](../rational-sigmoid-app-amd/README.md).','',
        f"Closure: `{pin(BASE/'closed.json')['sha256']}`.",
        'Raw evidence: `artifacts/parakeet-rational-sigmoid-models-amd-20260927`.']
    paths[1].write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n',encoding='utf8')
    paths[0].write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(passed=True,totals=payload['totals'],numeric_rows=rows,current_comparisons=cross,
                         samples=samples,peak_rss=peak,closure=payload['closure'])))


if __name__ == '__main__': main()
