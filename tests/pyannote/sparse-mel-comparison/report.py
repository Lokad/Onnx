"""Report every fixed-control verdict from the independent comparison closure."""
import json
from common import *


def main():
    closure, analysis = read(BASE / 'closed.json'), read(BASE / 'analysis.json')
    assert closure['passed'] and analysis['passed'] and closure['analysis'] == pin(BASE / 'analysis.json')
    verify(closure['files'])
    for name, wanted in closure['external_files'].items():
        assert pin(Path(name)) == wanted
    for identity in closure['terminal_identities']:
        terminal(identity)
    report, observations = TOOLS / 'results-20260921.md', TOOLS / 'observations-20260921.json'
    assert not report.exists() and not observations.exists()
    save(observations, dict(closure=pin(BASE / 'closed.json'), **analysis))
    table = '\n'.join(f"| {r['name']} | {r['seconds']['predecessor']:.6f} | {r['seconds']['candidate']:.6f} | {r['seconds']['ort']:.6f} | {r['candidate_to_predecessor']:.6f} | {r['candidate_to_ort']:.6f} |" for r in analysis['table'])
    controls = '\n'.join(f"| {role} | {name} | {ratio:.6f} | {'pass' if ratio <= (1.10 if name == 'dialogue-30s' else 1.20) else 'FAIL'} |" for role, row in analysis['controls'].items() for name, ratio in row['fixture_max_min'].items())
    admitted = analysis['qualifies_for_later_amd']
    if admitted:
        verdict = 'All fixed repeatability controls and performance admission gates pass. This candidate qualifies for later AMD evaluation; it is not a calibrated parity or production-promotion result.'
    elif analysis['attribution_valid']:
        verdict = 'Repeatability controls pass, but performance admission fails. Retain every observation; this trial does not select the candidate for later AMD performance evaluation.'
    else:
        verdict = 'Fixed repeatability controls fail. The observed means establish no timing admission, speedup or AMD candidate selection. Retain every observation and do not retry this unchanged trial.'
    report.write_text(f'''# Sparse mel: complete predecessor/candidate/ORT comparison

{verdict}

All {analysis['calls']} public requests and {analysis['resource_samples']:,} resource samples pass.
There are {analysis['warmup_calls']} warmups and {analysis['measured_calls']} measured calls;
no observations are removed. Public decisions, inputs and held outputs pass the
original native and ownership checks.

| Workload | Predecessor seconds | Candidate seconds | Microsoft ORT seconds | Candidate / predecessor | Candidate / ORT |
|---|---:|---:|---:|---:|---:|
{table}

Six fresh processes run predecessor, candidate, ORT, ORT, candidate,
predecessor. Each performs one complete warmup and three measured passes over
four fixtures. Windows i7-14700KF, CPU2, one CLR processor, normal .NET10.0.12,
Microsoft ORT1.29.0, NumPy2.2.4. The original complete AudioBenchmark and native
adapter include frontend processing, neural execution, clustering and owned
results; loading, file access and external validation are outside both timers.

Both managed roles use byte-identical Core5c0ae2aa. Data1d346664 is the accepted
dense predecessor and Datae9e4c28e the sparse candidate. All other runtime files
match. Full [application qualification](../sparse-mel-qualification/results-20260921.md)
is required before this comparison. The earlier vector-bias change is excluded.

| Role | Fixture | Process mean max/min | Fixed control |
|---|---|---:|---|
{controls}

Prospective limits remain 1.10 for the full request and 1.20 for every fixture.
Admission additionally requires full candidate/predecessor at most0.97 and
each fixture at most1.05. Failed controls are never repaired by deleting calls,
changing limits, calibrating means or combining historical comparisons.

Peak sampled worker RSS is {analysis['peak_rss']:,} bytes. All
{len(closure['terminal_identities'])} recorded identities are terminal. Original
preflight, available-memory, RSS, disk, output and time bounds pass. Allocations
and every individual observation are retained in
[observations-20260921.json](observations-20260921.json).

Artifact: `artifacts/pyannote-sparse-mel-comparison-20260921`.
Analysis SHA256: `{pin(BASE / 'analysis.json')['sha256']}`.
Closure: {pin(BASE / 'closed.json')['bytes']:,} bytes,
`{pin(BASE / 'closed.json')['sha256']}`.
Reproduction uses prepare.py with the exact successful application closure
digest, run.py, then audit.py after actual termination. The finite continuation
owns the current execution; existing tools and results remain immutable.
The primary AMD payload is unchanged by this Windows comparison.
''', encoding='utf8')
    print(json.dumps(dict(report=str(report.relative_to(ROOT)), admitted=admitted, closure=pin(BASE / 'closed.json'))))


if __name__ == '__main__':
    main()
