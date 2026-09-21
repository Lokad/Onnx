"""Summarize the closed complete application qualification and all allocations."""
import statistics
from common import *


def main():
    report, observations = TOOLS / 'results-20260922.md', TOOLS / 'observations-20260922.json'
    assert not report.exists() and not observations.exists()
    closure, analysis = read(BASE / 'closed.json'), read(BASE / 'analysis.json')
    assert closure['passed'] and analysis['passed'] and closure['analysis'] == pin(BASE / 'analysis.json')
    verify(closure['files'])
    for identity in closure['identities']:
        terminal(identity)
    old = read(PRIOR / 'analysis.json')
    allocations = []
    for name in dict.fromkeys(r['name'] for r in analysis['dialogue']):
        before = statistics.fmean(r['allocated_bytes'] for r in old['dialogue'] if r['name'] == name and r['phase'] == 'measured')
        after = statistics.fmean(r['allocated_bytes'] for r in analysis['dialogue'] if r['name'] == name and r['phase'] == 'measured')
        allocations.append(dict(name=name, predecessor_bytes=before, candidate_bytes=after, ratio=after / before))
    meetings = []
    for row, prior in zip(analysis['meetings'], old['meetings'], strict=True):
        assert row['name'] == prior['name']
        meetings.append(dict(name=row['name'], predecessor_bytes=prior['allocated_bytes'], candidate_bytes=row['allocated_bytes'],
            ratio=row['allocated_bytes'] / prior['allocated_bytes'], seconds=row['seconds']))
    save(observations, dict(closure=pin(BASE / 'closed.json'), **analysis, allocation_comparison=allocations, meeting_allocations=meetings))
    table = '\n'.join(f"| {r['name']} | {r['predecessor_bytes']:,.0f} | {r['candidate_bytes']:,.0f} | {r['ratio']:.4f} |" for r in allocations + meetings)
    report.write_text(f'''# Deferred views: complete public qualification

All **16 dialogue requests**, **both ten-minute meetings** and the **30-second
recovery request** pass with exact predecessor outputs. Ordinary and exclusive
speaker timelines match retained Microsoft ORT references exactly. Inputs and
held outputs remain unchanged. This qualifies correctness and ownership;
the [fresh timing comparison](../deferred-views-comparison/results-20260922.md)
has its own fixed controls and admission decision.

The candidate changes only RunTiledBatchFloat to construct tensor wrappers
after the specialized span-based matrix helper refuses. Its
[source and graph qualification](../deferred-views/results-20260922.md) includes
complete suites, hardware-disabled fallback, all captured outputs and affected
shared-model/Parakeet trajectories. Product instructions outside that method
and checked public declarations are unchanged.

## Complete-request allocation counters

| Workload | Prior qualified run bytes | Candidate bytes | Ratio |
|---|---:|---:|---:|
{table}

Dialogue entries average the three measured calls. Meeting/recovery entries
are the complete individual requests. These cumulative allocation counters
include runtime activity and are compared with the retained exact predecessor
qualification; no call is discarded. They are neither RSS nor matched latency
measurements. The fresh comparison measures predecessor, candidate and ORT
again before any speedup is admitted.

The original AudioBenchmark and NaturalMeetings consumers, models, PCM and
native references remain fixed. Windows i7-14700KF, CPU2, normal.NET10.0.12.
Core `{CORE}`; Data `{DATA}`. The original native centroid bounds are retained.
Every numerical and allocation observation is in
[observations](observations-20260922.json).

All **{analysis['resource_samples']:,} resource samples** and **{len(analysis['identities'])}
process identities** pass the independent audit. Peak sampled RSS is
{analysis['peak_rss']:,}bytes. Preflight10GiB,aggregateRSS8GiB,minimumavailable1GiB,
freedisk20GiB,outputs1GiB;dialogue900seconds and meeting stage3600seconds.
The actual application controller exits before audit/comparison begins.

Artifact: `artifacts/pyannote-deferred-views-applications-20260922`.
Closure: `{pin(BASE / 'closed.json')['sha256']}` ({pin(BASE / 'closed.json')['bytes']:,}bytes).
Reproduction uses run.py,then audit.py after actual termination;existing outputs
are refused. The finite comparison continuation owns the current transition;
do not run a duplicate audit or inference. Root production and primaryAMD
payload remain unchanged by this local qualification.
''', encoding='utf8')
    print(json.dumps(dict(report=str(report), allocations=allocations, meetings=meetings)))


if __name__ == '__main__':
    main()
