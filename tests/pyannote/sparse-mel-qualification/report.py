"""Export only the independently closed complete application result."""
from common import *


def main():
    closure, analysis = read(BASE / 'closed.json'), read(BASE / 'analysis.json')
    assert closure['passed'] and analysis['passed'] and analysis['exact_predecessor_results']
    assert closure['analysis'] == pin(BASE / 'analysis.json')
    verify(closure['files'])
    for identity in closure['identities']:
        terminal(identity)
    report, observations = TOOLS / 'results-20260921.md', TOOLS / 'observations-20260921.json'
    assert not report.exists() and not observations.exists()
    save(observations, dict(closure=pin(BASE / 'closed.json'), **analysis))
    suites = '\n'.join(f"| {r['name']} | {r['counters']['passed']} | {r['counters']['notExecuted']} |" for r in analysis['suites'])
    meetings = '\n'.join(f"| {r['name']} | {r['seconds']:.6f} | {r['maximum_centroid_error']:.9g} | {r['allocated_bytes']:,} |" for r in analysis['meetings'])
    report.write_text(f'''# Complete sparse-mel application qualification

The candidate passes all **16 complete dialogue requests**, both **600-second
meetings** and **30-second recovery**, with exact predecessor public results.
Ordinary and exclusive meeting timelines also match Microsoft ORT exactly.
Input and held-output ownership checks pass. These are correctness and resource
results; matched latency is measured separately.

| Suite | Passed | Retained skips |
|---|---:|---:|
{suites}

The independent auditor checks individual TRX outcomes as well as counters.
The suite copy uses exact Core5c0/Datae9 assemblies and the previously qualified
CLI pipe-draining fix. No product is rebuilt during application qualification.
The 58 new sparse-mel cases compare against the exact dense Data predecessor.
Core is byte-identical; its existing shared-model and Parakeet evidence is
inherited, retaining the three known native Parakeet duration failures.

| Workload | Observed seconds | Maximum centroid error versus ORT | Allocated bytes |
|---|---:|---:|---:|
{meetings}

Meeting times are observations without matched contemporary controls; they do
not establish a speedup. Allocations are cumulative managed bytes, not resident
memory, and are descriptive for this computation change. Full per-call evidence
is retained in [observations-20260921.json](observations-20260921.json).

All {analysis['resource_samples']:,} resource samples pass; all {len(analysis['identities'])}
recorded process identities are terminal. Peak sampled owned RSS is
{analysis['peak_rss']:,} bytes. Windows i7-14700KF, CPU2, normal .NET10.0.12.
Bounds remain 8/10GiB preflight, 8GiB owned RSS, 1GiB minimum available,
20GiB disk, 1GiB output, 900seconds for short children and 3600seconds for the
meeting consumer. Native accuracy limits are unchanged.

The [frontend proof](../sparse-mel/results-20260921.md) records the implementation,
89 normal/89 hardware-disabled tests and 4.312 million bit-identical real-audio
values. Only coefficient support changes; coefficients and reduction order stay
fixed. This candidate contains neither the unadmitted vector-bias change nor
the separately staged AMD AVX-512 route.

Core SHA256: `{CORE}`.
Data SHA256: `{DATA}`.
Artifact: `artifacts/pyannote-sparse-mel-applications-20260921`.
Analysis SHA256: `{pin(BASE / 'analysis.json')['sha256']}`.
Closure: {pin(BASE / 'closed.json')['bytes']:,} bytes,
`{pin(BASE / 'closed.json')['sha256']}`.

Run applications.py, reap its actual exit, then audit.py using
`C:/Python313/python.exe -X utf8 -B`. The already-launched finite comparison
continuation owns the current audit and timing stages; do not duplicate them.
Existing evidence is immutable. AMD performance and production integration
remain separate requirements.
''', encoding='utf8')
    print(json.dumps(dict(report=rel(report), closure=pin(BASE / 'closed.json'))))


if __name__ == '__main__':
    main()
