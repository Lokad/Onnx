"""Export the closed focused/frontend evidence without a latency claim."""
from common import *

FRONTEND = ROOT / 'artifacts/pyannote-sparse-mel-frontend-20260921'


def main():
    receipts = [(BASE / 'focused-closed.json', '35cba7a867ee2faff0fb8c3b94431c6f3ffa390648242188da2cd6515856cb6b'),
                (FRONTEND / 'closed.json', '71ac5283b789ff9ea4c8e55f18b12c54352656b020b323eb75dc4fc63a86a40d')]
    for path, sha in receipts:
        assert pin(path)['sha256'] == sha
        proof = read(path)
        assert proof['passed']
        verify(proof['files'])
        for identity in proof['identities']:
            terminal(identity)
    focused = read(BASE / 'focused-analysis.json')
    frontend = read(FRONTEND / 'analysis.json')
    instructions = read(BASE / 'instructions.json')
    assert instructions['passed']
    coefficients = frontend['coefficients']
    assert coefficients['dense_terms_per_frame'] == 20480
    assert coefficients['retained_terms_per_frame'] == coefficients['nonzero_terms_per_frame'] == 501
    observations = TOOLS / 'observations-20260921.json'
    report = TOOLS / 'results-20260921.md'
    patch = TOOLS / 'candidate.patch'
    assert not any(p.exists() for p in [observations, report, patch])
    patch.write_bytes((BASE / 'candidate.patch').read_bytes())
    save(observations, dict(closures=[dict(path=rel(path), **pin(path)) for path, _ in receipts],
        focused=focused, frontend=frontend,
        instructions=[{k:v for k,v in row.items() if k not in ['normalized_methods', 'candidate_methods']} for row in instructions['observations']],
        core=pin(BASE / 'runtime/Lokad.Onnx.dll'), data=pin(BASE / 'runtime/Lokad.Onnx.Data.dll'), patch=pin(patch)))
    report.write_text(f'''# Exact sparse mel frontend qualification

The isolated Data candidate skips leading and trailing zero weights in each
WeSpeaker mel band. It visits **501 of 20,480 coefficients per frame** while
keeping the original ascending double accumulation. The coefficient values,
FFT, powers, logarithm, centering, input checks, cancellation and ownership
remain unchanged. This reduces work in one frontend loop; it is not a measured
application speedup.

The [implementation patch](candidate.patch) adds private immutable support
ranges and one helper. Core is byte-identical to accepted Core5c0ae2aa, with all
3,107 methods unchanged. Data changes only LogMelFilterbank and its static
constructor, adds CreateMelSupport, and preserves the other 694 methods.
It starts from accepted Data1d346664; the unadmitted vector-bias experiment
is excluded. There is no new public API or dependency.

| Qualification | Result |
|---|---:|
| Normal frontend tests | 89 passed |
| Hardware-disabled frontend tests | 89 passed |
| Real-audio dense/sparse pairs | 50 exact |
| Real-audio frontend calls | 100 |
| Compared feature values | 4,312,000 bit-identical |
| Retained coefficients per frame | 501 / 20,480 |

Each focused mode includes 58 new cases and the 31 existing frontend cases.
Seven lengths from 400 to 480,000 samples cover frame boundaries, silence,
positive/negative extrema, alternating extrema, bounded noise, subnormal,
low-amplitude and signed-zero inputs. Checks cover coefficient support,
concurrent independent outputs, input preservation and held-output ownership.
Existing tests retain native direct-Fourier references, rejected inputs,
trailing-sample validation and cancellation.

Two sequential CPU2 workers reverse dense/sparse call order. Each processes
the four retained dialogue inputs and all 21 ten-second windows of the full
30-second dialogue at one-second offsets. Every emitted feature byte matches
the exact dense predecessor, and all inputs/held outputs remain unchanged.
The runtime coefficient digest is `{coefficients['coefficient_sha256']}`.
All retained intervals contain nonzero coefficients only. The original finite
normalized PCM bounds keep powers finite; skipping zero terms preserves the
nonnegative energy sum. Exact tests supplement that arithmetic reasoning.

The focused phase records {focused['resource_samples']} passing resource samples and
{len(focused['identities'])} terminal identities; frontend preparation/replay records
{frontend['resource_samples']} passing samples and {len(frontend['identities'])} terminal identities.
Windows i7-14700KF, CPU2, normal .NET10.0.12 except the explicit hardware-off
test mode. No model inference or latency comparison is claimed by this report.

Core SHA256: `{pin(BASE / 'runtime/Lokad.Onnx.dll')['sha256']}`.
Data SHA256: `{pin(BASE / 'runtime/Lokad.Onnx.Data.dll')['sha256']}`.
Focused closure: `{receipts[0][1]}`.
Real-audio closure: `{receipts[1][1]}`.
Full observations are retained in [observations-20260921.json](observations-20260921.json).

Reproduction uses prepare.py, then audit_preparation.py after actual exit,
followed by ../sparse-mel-frontend/prepare.py, run.py and audit.py sequentially
with `C:/Python313/python.exe -X utf8 -B`. Existing artifacts are immutable;
successors require new pinned directories. Complete public diarization,
meeting and matched ORT qualification are separate prerequisites for promotion.
The frozen AMD payload and current accepted benchmark remain unchanged.
''', encoding='utf8')
    print(json.dumps(dict(report=rel(report), observations=pin(observations))))


if __name__ == '__main__':
    main()
