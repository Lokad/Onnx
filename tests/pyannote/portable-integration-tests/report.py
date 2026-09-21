"""Export the closed test result and a single reviewable source integration patch."""
import difflib
import subprocess
from common import *


def main():
    proof, analysis = read(BASE / 'closed.json'), read(BASE / 'analysis.json')
    assert proof['passed'] and analysis['passed'] and proof['analysis'] == pin(BASE / 'analysis.json')
    verify(proof['files'])
    for identity in proof['identities']:
        terminal(identity)
    report = TOOLS / 'results-20260922.md'
    observations = TOOLS / 'observations-20260922.json'
    patch = TOOLS / 'integration.patch'
    assert not any(path.exists() for path in [report, observations, patch])
    source = BASE / 'source'
    wanted = sorted(read(PRIOR / 'source-inputs.json')['changed_text_files'] + ['tests/Lokad.Onnx.Backend.Tests/DenseWeSpeakerReference.cs'])
    parts, rows = [], []
    for name in wanted:
        target, original = source / name, ROOT / name
        before = original.read_text(encoding='utf-8-sig') if original.exists() else ''
        after = target.read_text(encoding='utf-8-sig')
        assert before != after, name
        header = 'diff --git a/' + name + ' b/' + name + '\n'
        if not original.exists():
            header += 'new file mode 100644\n'
        parts.append(header + ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
            fromfile='a/' + name if original.exists() else '/dev/null', tofile='b/' + name)))
        rows.append(dict(path=name, before=pin(original) if original.exists() else None, after=pin(target)))
    patch.write_text(''.join(parts), encoding='utf8', newline='\n')
    check = subprocess.run(['git', 'apply', '--check', str(patch)], cwd=ROOT, capture_output=True, text=True, check=True)
    assert not check.stdout and not check.stderr
    root_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    save(observations, dict(closure=pin(BASE / 'closed.json'), root_commit=root_commit, patch=pin(patch),
        apply_check_passed=True, patch_files=rows, **analysis))
    report.write_text(f'''# Self-contained tests for portable pyannote integration

The frontend tests now compile an internal copy of the original dense frontend
instead of loading an old Data DLL from ignored artifacts. Only that reference's
namespace and class declaration change; all original numerical and ownership
assertions remain intact. The existing native-reference frontend tests also run.

| Suite | Passed | Existing skips |
|---|---:|---:|
| Frontend | 89 | 0 |
| Frontend, hardware intrinsics disabled | 89 | 0 |
| Complete backend | 3,290 | 93 |
| Complete tensors | 342 | 0 |

The backend test assembly alone is rebuilt using the ordinary project references
and `BuildProjectReferences=false`. Every product source and binary remains
byte-identical to the [qualified normal build and package](../portable-integration/results-20260922.md).
No historical `dense-reference` directory is present before or after the tests.
The audit verifies individual TRX outcomes, including all 93 skipped cases.

Core: `{analysis['core']['sha256']}`.
Data: `{analysis['data']['sha256']}`.
All {analysis['resource_samples']} resource samples pass; {len(analysis['identities'])} process identities are terminal.
Peak sampled owned RSS is {analysis['peak_rss']:,} bytes. Windows CPU2, .NET10.0.12,
8/10GiB preflight,8GiB RSS ceiling,1GiB available,20GiB disk,1GiB output and900s
per child are unchanged. Controller session48214 exited0.

The [integration patch](integration.patch) combines all twelve product changes
with eleven test files, including the source-contained dense reference and the
qualified LSTM storage guard. It passes `git apply --check` against root commit
`{root_commit}`. No project declarations or unrelated source files are changed.
The patch is review material; root product sources are not yet modified.
AMD selection and any later combined dispatch qualification remain required.

This milestone adds no inference timing or new package. Accepted BENCHMARK
measurements retain their original binary identities. The subsequent integrated
application check is recorded separately in `../portable-applications`.

Closure: `{pin(BASE / 'closed.json')['sha256']}` ({pin(BASE / 'closed.json')['bytes']:,} bytes).
Patch: `{pin(patch)['sha256']}` ({pin(patch)['bytes']:,} bytes).
[Full observations](observations-20260922.json) include source pins, suite counters,
resource totals and the exact patch inputs. Preparation and audit refuse existing
outputs; retained evidence is never overwritten.
''', encoding='utf8')
    print(json.dumps(dict(report=rel(report), patch=pin(patch), changed_files=len(rows), apply_check_passed=True)))


if __name__ == '__main__':
    main()
