"""Summarize the independent application closure without assigning historical speedups."""
from common import *
import collections
import xml.etree.ElementTree as ET

BASE = ROOT / 'artifacts/pyannote-vector-bias-applications-20260921'


def main():
    closed = read(BASE / 'closed.json')
    analysis = read(BASE / 'analysis.json')
    assert closed['passed'] and analysis['passed'] and analysis['exact_predecessor_results']
    assert pin(BASE / 'analysis.json') == closed['analysis']
    for identity in closed['identities']:
        terminal(identity)
    assert all(row['exact_predecessor'] and row['exact_native_timelines'] for row in analysis['meetings'])
    suites = []
    for row in analysis['suites']:
        path = BASE / 'test-results' / (row['name'] + '.trx')
        assert pin(path) == closed['files'][rel(path)]
        tree = ET.parse(path)
        assert tree.find('.//{*}Counters').attrib == row['counters']
        outcomes = collections.Counter(r.attrib['outcome'] for r in tree.findall('.//{*}UnitTestResult'))
        assert outcomes['Passed'] == int(row['counters']['passed']) and not outcomes['Failed']
        suites.append(dict(name=row['name'], outcomes=dict(outcomes)))
    observations = TOOLS / 'observations-20260921.json'
    report = TOOLS / 'results-20260921.md'
    assert not observations.exists() and not report.exists()
    save(observations, dict(closure=pin(BASE / 'closed.json'), verified_suite_outcomes=suites, **analysis))
    suite_rows = '\n'.join(f"| {r['name']} | {r['outcomes'].get('Passed', 0)} | {r['outcomes'].get('NotExecuted', 0)} |" for r in suites)
    meeting_rows = '\n'.join(f"| {r['name']} | {r['seconds']:.6f} | {r['maximum_centroid_error']:.9g} | {r['allocated_bytes']:,} | {r['allocation_ratio']:.6f} |" for r in analysis['meetings'])
    dialogue_rows = '\n'.join(f"| {r['name']} | {r['prior_allocated_mean']:,.0f} | {r['candidate_allocated_mean']:,.0f} | {r['allocation_ratio']:.6f} |" for r in analysis['dialogue'])
    report.write_text(f'''# Complete vector-bias application qualification

The candidate passes the full backend/tensor suites, all **16 public dialogue
requests**, both **600-second natural meetings** and a **30-second recovery**.
Every public result exactly matches the immediate Core5c0ae2aa predecessor.
Ordinary and exclusive meeting timelines also exactly match Microsoft ORT.
These are correctness and resource results; fresh matched timing is separate.

| Suite | Passed | Retained skips |
|---|---:|---:|
{suite_rows}

TRX individual outcomes are checked independently of its aggregate counters;
no skipped test is described as passed. The suite copy retains the proven
concurrent CLI test-pipe draining fix, with unchanged timeout and assertions.
CLI/backend/tensor consumers reference the exact frozen candidate runtime;
there is no product rebuild during application qualification.

The [implementation report](../vector-bias/results-20260921.md) describes the
single changed convolution method, 358 focused/144 hardware-disabled tests,
108 graph calls preserving 17.5 million values, shared-model native checks and
exact Parakeet regression. The three original Parakeet native numerical
failures remain explicit. Neither these regressions nor the historical
predecessor's 6.3% improvement supplies this candidate's speed verdict.

## Natural meetings and allocation observations

The original NaturalMeetings consumer executes ES2004a and IS1009a, then the
recovery request. Inputs and held outputs remain unchanged. Identical speaker
timelines preserve the retained meeting-quality scores.

| Workload | Observed seconds | Maximum centroid error versus ORT | Allocated bytes | Allocation / predecessor |
|---|---:|---:|---:|---:|
{meeting_rows}

| Dialogue | Predecessor allocated mean | Candidate allocated mean | Ratio |
|---|---:|---:|---:|
{dialogue_rows}

These are cumulative managed allocation counters, not resident memory. They
are descriptive and are not an admission requirement for this computation
change. Historical predecessor times are not matched controls; do not infer
a speedup from the meeting observations.

## Resource and closure evidence

All {analysis['resource_samples']:,} resource samples pass; all {len(analysis['identities'])} recorded identities are terminal.
Maximum sampled process-group RSS is {analysis['peak_rss']:,} bytes across suites and
public qualification. Windows i7-14700KF, CPU2 and normal .NET10.0.12 settings
match the protocol. Bounds remain 8/10GiB preflight, 8GiB owned RSS, 1GiB minimum
available, 20GiB disk, 1GiB output, 900seconds for short children and 3600seconds
for the complete meeting consumer. All original native bounds remain fixed.

Core SHA256: `{CORE}`.
Data SHA256: `{DATA}`.
Artifact: `artifacts/pyannote-vector-bias-applications-20260921`.
Analysis SHA256: `{pin(BASE / 'analysis.json')['sha256']}`.
Closure: {pin(BASE / 'closed.json')['bytes']:,} bytes,
`{pin(BASE / 'closed.json')['sha256']}`, pinning {len(closed['files']):,} files.

Application session3115/PID246908birth1790023793.9550571 is terminal with exit0.
The finite continuation owns and has completed the independent application
audit. Its next stages prepare/run/audit the new six-process predecessor/
candidate/ORT comparison. Do not duplicate these stages or change frozen tools.
Production promotion and AMD performance remain pending.

Reproduction uses applications_v2.py followed, after actual termination, by
audit_applications_v2.py with `C:/Python313/python.exe -X utf8 -B`. Existing
outputs are immutable; new attempts require separate pinned successors.
''', encoding='utf8')
    print(json.dumps(dict(report=rel(report), closure=pin(BASE / 'closed.json'))))


if __name__ == '__main__':
    main()
