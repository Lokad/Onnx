"""Publish the already-closed comparison without changing any measured evidence."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'artifacts/pyannote-convolution-portable-comparison-20260921'
HERE = Path(__file__).resolve().parent


def pin(path):
    data = path.read_bytes()
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def main():
    closed = json.loads((BASE / 'closed.json').read_text())
    for name, expected in closed['files'].items():
        assert pin(ROOT / name) == expected, name
    for name, expected in closed['external_files'].items():
        assert pin(Path(name)) == expected, name
    analysis = json.loads((BASE / 'analysis.json').read_text())
    assert pin(BASE / 'analysis.json') == closed['analysis']
    assert analysis['passed'] and analysis['attribution_valid'] and analysis['qualifies_for_later_amd']
    observations = HERE / 'observations-20260921.json'
    report = HERE / 'results-20260921.md'
    assert not observations.exists() and not report.exists()
    observations.write_text(json.dumps(dict(closure=pin(BASE / 'closed.json'), **analysis), indent=2) + '\n', encoding='utf8')
    rows = '\n'.join(
        f"| {r['name']} | {r['seconds']['predecessor']:.6f} | {r['seconds']['candidate']:.6f} | {r['seconds']['ort']:.6f} | {100*(1-r['candidate_to_predecessor']):.2f}% | {r['candidate_to_ort']:.6f} |"
        for r in analysis['table'])
    controls = '\n'.join(f"| {role} | " + ' | '.join(f'{value:.6f}' for value in data['fixture_max_min'].values()) + ' |'
                         for role, data in analysis['controls'].items())
    workers = '\n'.join(f"| {w['index']} | {w['role']} | " + ' | '.join(f'{value:.6f}' for value in w['means'].values()) + ' |'
                        for w in analysis['workers'])
    report.write_text(f'''# Portable convolution: complete application comparison

The candidate reduces the complete 30-second pyannote request from **12.325453
to 11.548528 seconds (6.30%)** in this fresh Windows comparison. Microsoft ORT
takes **6.333994 seconds**, leaving a **1.823262 candidate/ORT ratio**. Every
fixed process-repeatability and candidate-admission control passes. These are
descriptive local measurements, without calibrated parity, AMD timing or
production promotion.

| Workload | Predecessor seconds | Candidate seconds | ORT seconds | Reduction | Candidate / ORT |
|---|---:|---:|---:|---:|---:|
{rows}

The predecessor is the qualified pooled-convolution Core0d098ba5; the candidate
is Core5c0ae2aa. Both use Data1d346664. Only tiled convolution row grouping
changes. The [implementation](../convolution-portable-rows/results-20260921.md)
preserves 17.5 million captured values and passes shared-model and affected
Parakeet regression. The [full application qualification](../convolution-portable-qualification/results-20260921.md)
passes 3,222 backend tests, 342 tensor tests, two request-focused tests, all
16 dialogue calls, two ten-minute meetings and recovery. Native meeting
timelines remain exact. The three pre-existing Parakeet native failures are
preserved; this candidate does not fix them.

## Protocol and controls

Windows i7-14700KF, logical CPU2, normal .NET 10.0.12 and Microsoft ORT 1.29.0.
Six fresh processes run predecessor/candidate/ORT/ORT/candidate/predecessor.
Each executes one warmup and three measured passes over four fixtures:
96 calls, 24 warmups and 72 measurements. Loading, file access and external
validation are outside both timers; feature extraction, neural inference,
clustering and owned result construction are inside. The original managed
consumer and native adapter are unchanged. All public/native, input and
retained-output checks pass; every observation is retained in the adjacent JSON.

Before execution, repeatability required each role's two full-request process
means to differ by at most 1.10, and each fixture by at most 1.20. Candidate
admission required full candidate/predecessor at most 0.97 and every fixture
at most 1.05. No threshold or fixture changed after measurement. Allocation
reduction was prospectively omitted as an entry requirement for this computation
change; all numerical, timing and resource requirements were retained.

| Role | Full 30s max/min | First crop | Second crop | Third crop |
|---|---:|---:|---:|---:|
{controls}

| Process | Role | Full 30s mean | First crop | Second crop | Third crop |
|---|---|---:|---:|---:|---:|
{workers}

All {analysis['resource_samples']:,} resource samples pass; peak sampled worker RSS is
{analysis['peak_rss']:,} bytes. Seven controller/worker identities are terminal.
Each child retains the 10 GiB preflight, 8 GiB owned RSS, 1 GiB available-memory,
20 GiB disk, 1 GiB output and 1,800-second limits. Cumulative full-request managed
allocations are 1,125,675,355 predecessor and 1,098,755,316 candidate bytes.
Those counters are descriptive and are not resident memory.

## Evidence and scope

This is a new implementation and fresh comparison after complete qualification,
not an unchanged retry of either earlier allocation candidate's failed timing
controls. Those failures remain intact. The original kernel probe's separate
[coverage correction](../portable-row-groups/coverage-correction-20260921.md)
also remains explicit: its 17.8% result covered 16 of 22 tile shapes. Complete
graph and public workloads here execute all layers; a separate complete-shape
kernel successor checks the missing per-shape behavior.

Core predecessor SHA256:
`0d098ba5fd3fd8799bb1dd018148123f296802c769db5f6148a1a5fa9d80118e`.
Core candidate SHA256:
`5c0ae2aa7c3cce58f3ffcb190df451e053a449a3e0dbc920d7b6d0b2bc66020c`.
Common Data SHA256:
`1d34666456a5da749dc3b40ee25621af0bed806c9167f3ad12e96d0736dca662`.
Artifact: `artifacts/pyannote-convolution-portable-comparison-20260921`.
Analysis: {pin(BASE / 'analysis.json')['bytes']:,} bytes,
`{pin(BASE / 'analysis.json')['sha256']}`.
Closure: {pin(BASE / 'closed.json')['bytes']:,} bytes,
`{pin(BASE / 'closed.json')['sha256']}`;
{len(closed['files'])} repository and {len(closed['external_files'])} external file pins.

The finite finish controller completed application audit, comparison preparation,
execution and audit, then was independently closed. Its closure SHA256 is
`108fa8aecb185a1c5a09191016ca5060394ba3b255ecde09861d3bfd9cef541a`.
Reproduction tools are prepare.py, run.py and audit.py here, with the application
closure passed to preparation. Use `C:/Python313/python.exe -X utf8 -B` from the
repository root. Existing outputs are immutable and tools refuse overwrites;
new runs require separate pinned successors. This report verifies all closed
file pins before writing. The frozen AMD payload and its current owners remain
unchanged.
''', encoding='utf8')
    print(json.dumps(dict(report=str(report), closure=pin(BASE / 'closed.json'))))


if __name__ == '__main__':
    main()
