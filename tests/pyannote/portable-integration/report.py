"""Summarize the closed normal build and package, retaining the checker incident."""
from resume import *


def main():
    closure, analysis = read(BASE / 'closed.json'), read(BASE / 'analysis.json')
    assert closure['passed'] and analysis['passed'] and closure['analysis'] == pin(BASE / 'analysis.json')
    verify(closure['files'])
    for identity in closure['identities']:
        terminal(identity)
    report, observations = TOOLS / 'results-20260922.md', TOOLS / 'observations-20260922.json'
    assert not report.exists() and not observations.exists()
    save(observations, dict(closure=pin(BASE / 'closed.json'), preserved_failure=pin(PRIOR / 'failure-closed.json'), **analysis))
    rows = '\n'.join(f"| {r['name']} | {r['outcomes'].get('Passed', 0)} | {r['outcomes'].get('NotExecuted', 0)} |" for r in analysis['suites'])
    report.write_text(f'''# Normal pyannote source build and local package qualification

The reviewed portable candidate builds through normal project references,
including the mandatory LSTM storage guard. Core/Data method inspection and
public-declaration comparison pass, followed by the complete local suites and
a separate application restored from the newly packed Core NuGet package.
This prepares source/package integration; it is not a new timing result or
production promotion.

| Suite | Passed | Retained skips |
|---|---:|---:|
{rows}

The source tree starts from tracked product files at `{analysis['root_commit']}`.
It overlays exactly twelve reviewed product files and ten additional test files.
Current project declarations and the CLI pipe-draining correction are retained.
No experiment-specific assembly HintPath is inserted into the product or its
backend/tensor projects. Fifty encoding-only differences are excluded.

All **697 Data methods** and **3,106 Core methods** match the accepted Core5c0/
Datae9 candidate. Only private LstmProjectionPanels.Create changes and its
StorageLength helper is added. The guard declines an unrepresentable optional
array using a wide sum before allocation, preserving the scalar fallback.
All **9,168 Core and 1,626 Data public-declaration records** match, including
the checked types, methods, constructors, fields, properties, events and
parameter defaults. There is no public API addition.

The package-only consumer has PackageReference to Lokad.Onnx0.2.0 and no project
or assembly references. Its loaded Core is byte-identical to the inspected
runtime and the nupkg's lib/net10.0/Lokad.Onnx.dll. Matrix, convolution, ConvRelu,
input/held-tensor checks and MNIST ONNX import through packaged protobuf pass.
Data remains non-packable under the repository's existing package contract;
the complete Data/CLI runtime is retained separately.

Core SHA256: `{analysis['core']['sha256']}`.
Data SHA256: `{analysis['data']['sha256']}`.
Package: {analysis['package']['bytes']:,} bytes,
`{analysis['package']['sha256']}`.
Local package path:
`artifacts/pyannote-portable-integration-completion-20260922/nuget/Lokad.Onnx.0.2.0.nupkg`.
The only package dependency is Google.Protobuf3.33.5, targetingnet10.0.

## Preserved checker refusal and continuation

The first controller exited after the backend suite because its checker expected
TRX's aggregate notExecuted counter to equal93. Individual outcomes correctly
show3,290Passed and93NotExecuted, while the aggregate notExecuted is0. No backend
test failed. The corrected reader verifies every outcome and total/executed/
passed counters, retaining the original raw aggregate fields.

Failure closure `{pin(PRIOR / 'failure-closed.json')['sha256']}` preserves that
attempt. Its completed builds, method/API proof, focused and backend suites
are reused without rerunning them. A separate continuation runs only tensors,
packing and package consumption. The independently closed total is
**{analysis['resource_samples']} resource samples**, **{len(analysis['identities'])} terminal process identities**,
with peak sampled owned RSS **{analysis['peak_rss']:,} bytes**. All original limits
hold:8/10GiBpreflight,8GiBRSS,1GiBavailable,20GiBdisk,1GiBoutput,900seconds per
child. Windows CPU2 and normal .NET10.0.12, except the named hardware-off suite.

Root input snapshots preserve the exact pre-integration files separately from
the candidate. Later authorized source edits need not rewrite this historical
evidence. The source patch is
`artifacts/pyannote-portable-integration-20260922/candidate.patch`.
The current SparseMelTests still loads the frozen dense Data reference supplied
by the qualification harness. Replace that test-only artifact dependency with
an equivalent self-contained source reference before final root integration.

Completion closure: {pin(BASE / 'closed.json')['bytes']:,} bytes,
`{pin(BASE / 'closed.json')['sha256']}`.
Full observations: [observations-20260922.json](observations-20260922.json).
Original tools prepare.py/common.py remain unchanged. Reproduction records
close_failure.py, resume.py and audit_completion.py; existing artifacts cannot
be overwritten. The root product, frozen AMD payload and accepted benchmark
binaries remain unchanged. Final application/target qualification and source
integration are still required.
''', encoding='utf8')
    print(json.dumps(dict(report=rel(report), closure=pin(BASE / 'closed.json'))))


if __name__ == '__main__':
    main()
