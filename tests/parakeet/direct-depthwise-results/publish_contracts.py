"""Publish the completed candidate contracts without rerunning qualification."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'artifacts/parakeet-direct-depthwise-build-v2-amd-20260925'
OUT = Path(__file__).resolve().parent


def read(p): return json.loads(p.read_text(encoding='utf8'))
def pin(p):
    with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())


def main():
    closed=read(BASE/'closed.json');analysis=read(BASE/'analysis.json');build=read(BASE/'build-review.json')
    assert closed['passed'] and closed['analysis']==pin(BASE/'analysis.json')
    assert analysis['passed'] and analysis['compiled_review']==pin(BASE/'build-review.json')
    correction=read(BASE/'scope-correction.json');assert build['scope_correction']==pin(BASE/'scope-correction.json')
    assert correction['passed'] and correction['proof']['all_instructions_locals_exceptions_and_flags_equal']
    source=ROOT/'artifacts/parakeet-direct-depthwise-source-v2-20260925'
    prepared=read(source/'prepared.json');assert analysis['source']==pin(source/'prepared.json')
    assert prepared['test_only_correction']
    initial=ROOT/'artifacts/parakeet-direct-depthwise-source-20260925'
    original=read(initial/'prepared.json');assert prepared['original_candidate']==pin(initial/'prepared.json')
    for name in original['source']:
        if name.startswith('src/'):assert prepared['source'][name]==original['source'][name]
    assert len(analysis['suites'])==2
    for suite in analysis['suites']:
        assert suite['passed']==8 and suite['skipped']==0
        assert suite['geometry']['geometries']==59 and suite['geometry']['checked_values']==57332736
    result=dict(passed=True,closure=pin(BASE/'closed.json'),compiled_review=pin(BASE/'build-review.json'),
                original_candidate=pin(initial/'prepared.json'),corrected_source=pin(source/'prepared.json'),
                analysis=analysis,compiled_scope=build['methods'],no_added_warnings=build['zero_added_warnings'],
                first_build_failure_preserved=True,test_only_correction=True,
                native_model_checks_pending=True,mechanism_capture_pending=True,performance_pending=True,release_admitted=False)
    with (OUT/'contracts-20260925.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
    lines=['# Direct Parakeet depthwise candidate: focused numerical qualification', '',
        '**All eight tests pass in both normal and hardware-disabled processes.**',
        'Each process checks **57,332,736 output values across all 59 actual',
        'geometries** against the separately loaded original M78 Core. Finite',
        'outputs and signed zeros agree bit-for-bit; explicit special-value tests',
        'also preserve NaN classification. No performance result follows from this.', '',
        'The candidate directly accumulates nine taps in the existing dense layout.',
        'It retains the original FMA positions, multiply/add tail and sum-then-bias',
        'order. Explicit segmented selection and unsupported geometry, batch, degree',
        'or hardware options keep their existing paths. Output allocation is unchanged.', '',
        'The tests cover full observed geometries, spatial borders and flattened',
        'vector tails, line tails and memory offsets, special values including',
        'padding, fallback options/geometries, logical views and independent outputs,',
        'the 1D provider, dirty pooled spatial outputs, and runtime/product identities.',
        'Eligible normal-hardware operator calls report zero scratch; complete',
        'application mechanism counters remain a separate requirement.', '',
        '| Compiled scope | Verified |', '| --- | ---: |',
        '| Changed original Core methods | 1 |', '| Added private helpers | 4 |',
        '| Other original Core methods unchanged after exact metadata reconciliation | 3,276 |',
        '| Data methods unchanged | 697 |', '| Added warnings | 0 |', '',
        'Data is reused byte-for-byte. Three compiler-generated private names shifted',
        'by four and two callers refer to those shifted names; every instruction,',
        'local, exception region and implementation flag otherwise matches exactly.',
        'The initial reviewer refusal and the additive reconciliation are retained.',
        'Three rejection tests cover instruction, flag and unexpected-name changes.', '',
        'The first build compiled Core but failed in the new test harness: generic',
        'type inference for TryGetArray and a platform-analysis warning. The corrected',
        'source changes only those two test expressions. Every product source byte',
        'matches the first candidate. The failed build remains recorded.', '',
        f"Candidate Core SHA256: `{analysis['product']['Lokad.Onnx.dll']['sha256']}`.",
        f"Data SHA256: `{analysis['product']['Lokad.Onnx.Data.dll']['sha256']}`.",
        f"Closure SHA256: `{result['closure']['sha256']}`.", '',
        'All 242 resource observations pass. Peak owned RSS is 1,699,147,776 bytes.',
        'Both workers and their owner are terminal with exit code zero. The full',
        'transcription mechanism capture, native-model correctness, shape-weighted',
        'screen and complete application/regression admission remain outstanding.',
        'M78’s independent e5 failure still prevents release promotion.', '',
        '[Per-geometry evidence and test names](contracts-20260925.json),',
        '[causal diagnosis](../depthwise-route-results/diagnosis-20260925.md).', '']
    with (OUT/'contracts-20260925.md').open('x',encoding='utf8') as f:f.write('\n'.join(lines))
    print(json.dumps(dict(report=pin(OUT/'contracts-20260925.md'),evidence=pin(OUT/'contracts-20260925.json'))))


if __name__=='__main__':main()
