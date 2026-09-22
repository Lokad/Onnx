"""Publish only the closed, admitted prepared-convolution source integration."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'tests/pyannote/blocked-spatial-root'))
from run import BASE, AMD, MODEL, NAMES, pin, read, verify, terminal


def main():
    target = TOOLS / 'results-20260922.md'
    assert not target.exists()
    closed = read(BASE / 'closed.json')
    assert closed['passed'] and closed['analysis'] == pin(BASE / 'analysis.json')
    verify(closed['files'])
    for identity in closed['identities']:
        terminal(identity)
    result = read(BASE / 'analysis.json')
    assert result['passed'] and result['no_new_performance_measurement']
    assert result['amd_closure'] == pin(AMD / 'closed.json')
    timing = read(AMD / 'analysis.json')
    assert timing['passed'] and timing['performance']['admitted']
    assert timing['timing_calls'] == 96 and timing['measured'] == 72
    snapshots = read(BASE / 'source-snapshots.json')
    assert set(snapshots) == set(NAMES)
    for name, value in snapshots.items():
        wanted = {k: value[k] for k in ['sha256', 'bytes']}
        assert pin(ROOT / name) == pin(ROOT / value['path']) == wanted
    consumer = read(BASE / 'consumer.json')
    assert consumer['passed'] and consumer['core'] == result['core']['sha256']
    assert consumer['prepared_graph_values'] == 1056 and consumer['prepared_graph_calls'] == 2
    assert consumer['retained_weights'] == 18432 and consumer['graph_scratch'] == 8384
    assert consumer['model_imported'] and consumer['input_and_held_outputs_unchanged']
    full = next(r for r in timing['table'] if r['audio_seconds'] == 30)
    gain = 100 * (1 - full['portable']['seconds'] / full['production']['seconds'])
    suites = '\n'.join(f"| {r['name']} | {r['passed']:,} | {r['skipped']:,} |" for r in result['suites'])
    peak = max(r['peak_rss'] for r in result['resources'])
    target.write_text(f'''# Prepared convolution integrated into normal root source

The selected prepared-convolution change is applied to the ordinary product
and test projects. Eligible constant 3x3 weights share the existing preparation
budget with MatMul. Guarded spatial kernels use blocked-channel scratch, preserve
ordinary graph fusion and return owned results. Unsupported shapes, policies,
hardware and nonfinite inputs retain the existing fallback. Public APIs and
package dependencies remain unchanged.

The [complete AMD comparison](../blocked-spatial-app-results/results-20260922.md)
measures the exact candidate at **{full['portable']['seconds']:.3f} s**, contemporary
production at **{full['production']['seconds']:.3f} s** and Microsoft ORT at
**{full['ort']['seconds']:.3f} s**: **{gain:.2f}% less latency**, with a
**{full['ratios_to_ort']['portable']:.3f} ratio to ORT**. All 96 requests, twelve
repeatability controls and four speed gates pass. The <=1.05 parity target
remains separate from this accepted improvement.

The normal root build matches all **3,161 Core and 697 Data methods** and public
declarations of measured Core `3c2f16b0` / Data `6318cf48`. Its own identities
are below; this source equivalence does not create a new rebuild timing claim.
The [isolated qualification](../blocked-spatial-composition-results/results-20260922.md)
and [complete Windows model checks](../blocked-spatial-model-results/results-20260922.md)
remain separately recorded.

| Suite | Passed | Skipped |
|---|---:|---:|
{suites}

Normal CLI/backend/tensor builds pass. The resulting NuGet package contains the
exact inspected Core and retains Google.Protobuf 3.33.5 as its only dependency.
An independent PackageReference consumer checks public operators, ConvRelu,
MNIST import and input/held-output ownership. It also verifies two prepared
ConvRelu graph requests with 1,056 output values each, 18,432 retained weight
bytes and exactly 8,384 requested scratch bytes.


All **{result['resource_samples']:,} resource observations** pass; peak owned RSS is
**{peak:,} bytes**. All {len(closed['identities'])} recorded process identities are
terminal. Windows CPU2, normal .NET 10.0.12; unchanged 8/12 GiB preflight,
8 GiB RSS, 1 GiB available/output, 20 GiB disk and 900-second worker limits.

Root Core: `{result['core']['sha256']}` ({result['core']['bytes']:,} bytes).
Root Data: `{result['data']['sha256']}` ({result['data']['bytes']:,} bytes).
Package: `{result['package']['sha256']}` ({result['package']['bytes']:,} bytes).
Closure: `{pin(BASE / 'closed.json')['sha256']}`.

Source snapshots and the package are retained in
`artifacts/pyannote-blocked-spatial-root-20260922`. Reproduction and admission checks
are in [the integration protocol](../blocked-spatial-root/README.md). No package or
branch is published by these tools. Parakeet remains the next audio priority;
Whisper work stays deferred.
''', encoding='utf8')
    print(json.dumps(dict(report=str(target.relative_to(ROOT)), core=result['core'],
                          data=result['data'], closure=pin(BASE / 'closed.json'))))


if __name__ == '__main__':
    main()
