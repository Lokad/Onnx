"""Report every closed observation and the original timing verdict, including failure."""
from common import *

FINISH = ROOT / 'artifacts/pyannote-vector-bias-comparison-finish-20260921'


def main():
    closed, analysis, spec = read(BASE / 'closed.json'), read(BASE / 'analysis.json'), read(BASE / 'prepared.json')
    finish = read(FINISH / 'closed.json')
    assert closed['passed'] and analysis['passed'] and finish['passed']
    assert pin(BASE / 'analysis.json') == closed['analysis']
    assert closed['attribution_valid'] == analysis['attribution_valid'] == finish['attribution_valid']
    verify(closed['files'])
    for name, expected in closed['external_files'].items():
        assert pin(Path(name)) == expected
    for identity in [*closed['terminal_identities'], *finish['terminal_identities']]:
        terminal(identity)
    full = analysis['table'][0]
    valid, admitted = analysis['attribution_valid'], analysis['qualifies_for_later_amd']
    if admitted:
        lead = f"The candidate passes every fixed timing control and admission gate. The full request falls from **{full['seconds']['predecessor']:.6f} to {full['seconds']['candidate']:.6f} seconds ({100*(1-full['candidate_to_predecessor']):.2f}% lower)**; Microsoft ORT takes **{full['seconds']['ort']:.6f} seconds**, leaving a **{full['candidate_to_ort']:.6f} ratio**."
    elif not valid:
        lead = 'The candidate passes all public-output and resource checks, but **fails the fixed timing repeatability controls**. It establishes no application speedup or timing admission. The accepted benchmark table remains unchanged.'
    else:
        lead = 'All fixed repeatability controls pass, but **the candidate fails the application improvement gate**. It is not admitted as a performance improvement. The accepted benchmark table remains unchanged.'
    table = '\n'.join(f"| {r['name']} | {r['seconds']['predecessor']:.6f} | {r['seconds']['candidate']:.6f} | {r['seconds']['ort']:.6f} | {r['candidate_to_predecessor']:.6f} | {r['candidate_to_ort']:.6f} |" for r in analysis['table'])
    controls = '\n'.join(f"| {role} | " + ' | '.join(f'{value:.6f}' for value in row['fixture_max_min'].values()) + f" | {row['passed']} |" for role, row in analysis['controls'].items())
    workers = '\n'.join(f"| {r['index']} | {r['role']} | " + ' | '.join(f'{value:.6f}' for value in r['means'].values()) + ' |' for r in analysis['workers'])
    observations, report = TOOLS / 'observations-20260921.json', TOOLS / 'results-20260921.md'
    assert not observations.exists() and not report.exists()
    save(observations, dict(closure=pin(BASE / 'closed.json'), finish_closure=pin(FINISH / 'closed.json'), **analysis))
    report.write_text(f'''# Vectorized convolution bias: complete application comparison

{lead}

| Workload | Predecessor seconds | Candidate seconds | ORT seconds | Candidate / predecessor | Candidate / ORT |
|---|---:|---:|---:|---:|---:|
{table}

These are retained descriptive Windows observations. They do not establish
calibrated parity, AMD timing or production promotion. A failed control prevents
using the corresponding point estimates as a gain. Every sample is retained;
there is no unchanged retry or retrospective threshold adjustment.

## Qualification and protocol

The [single-method implementation](../vector-bias/results-20260921.md) preserves
17.5 million graph values, all shared-model native arrays and all affected
Parakeet arrays. The [complete application qualification](../vector-bias-qualification/results-20260921.md)
passes 3,366 backend tests, 342 tensor tests, two request-focused tests, all16
dialogue calls, both ten-minute meetings and recovery. Predecessor outputs and
native meeting timelines are exact. The three pre-existing Parakeet native
numerical failures remain unchanged; this candidate does not fix them.

Windows i7-14700KF, CPU2, normal .NET10.0.12 and Microsoft ORT1.29.0. Six fresh
workers run predecessor/candidate/ORT/ORT/candidate/predecessor. Each performs
one warmup and three measured passes over four fixtures:96calls,24warmups and
72measurements. Original managed/native consumers and accuracy/input/ownership
checks remain unchanged. Loading, file access and external validation are outside
both timers; features, graphs, clustering and owned result creation are inside.

The predeclared process max/min limits are1.10for the full request and1.20for
each fixture in each role. Admission also requires full candidate/predecessor
at most.97and every fixture at most1.05. Allocation counters are descriptive;
their reduction was prospectively not an entry requirement for this computation
candidate. All output and resource requirements remain fixed.

| Role | Full max/min | First crop | Second crop | Third crop | Controls pass |
|---|---:|---:|---:|---:|---|
{controls}

| Process | Role | Full30s mean | First crop | Second crop | Third crop |
|---|---|---:|---:|---:|---:|
{workers}

All {analysis['calls']} requests and {analysis['resource_samples']:,} resource samples pass. Peak sampled worker
RSS is {analysis['peak_rss']:,} bytes; all {len(closed['terminal_identities'])} recorded controller/worker identities
are terminal. Bounds remain10GiBpreflight,8GiBownedRSS,1GiBavailable,20GiBdisk,
1GiBoutput and1800seconds per child. Complete allocations, GC readings, warmup
and measured timings remain in the adjacent observations JSON.

## Evidence

Predecessor Core SHA256:
`{spec['roles']['predecessor']['Lokad.Onnx.dll']['sha256']}`.
Candidate Core SHA256:
`{spec['roles']['candidate']['Lokad.Onnx.dll']['sha256']}`.
Common Data SHA256:
`{spec['roles']['candidate']['Lokad.Onnx.Data.dll']['sha256']}`.
Artifact: `artifacts/pyannote-vector-bias-comparison-20260921`.
Analysis: {pin(BASE / 'analysis.json')['bytes']:,} bytes,
`{pin(BASE / 'analysis.json')['sha256']}`.
Closure: {pin(BASE / 'closed.json')['bytes']:,} bytes,
`{pin(BASE / 'closed.json')['sha256']}`;
{len(closed['files'])} repository and {len(closed['external_files'])} external file pins.
Independent finite-continuation closure:
`{pin(FINISH / 'closed.json')['sha256']}`.

The application controller and continuation are terminal, with all four audit/
preparation/execution/audit stages completed. Reproduction tools are prepare.py,
run.py and audit.py here; preparation takes the completed application closure
SHA256. Use `C:/Python313/python.exe -X utf8 -B` from the repository root.
Existing evidence and executed tools are immutable; any new hypothesis requires
a separately pinned successor. The current frozen AMD payload remains unchanged.
''', encoding='utf8')
    print(json.dumps(dict(report=str(report), attribution_valid=valid, qualifies_for_later_amd=admitted)))


if __name__ == '__main__':
    main()
