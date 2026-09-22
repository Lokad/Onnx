"""Publish the complete direct-output comparison without changing its verdict."""
from fractions import Fraction
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'tests/pyannote/direct-amd-v2'))
from transport import BASE, SITE, checked_local
from candidate_protocol import pin, read, verified_files, ROLES, TIMING_ROLES, gate
from admission import evaluate, ROLE_LABELS
sys.path.insert(0, str(SITE))
import psutil


def main():
    report = TOOLS / 'results-20260922.md'
    observations = TOOLS / 'observations-20260922.json'
    assert not report.exists() and not observations.exists()
    checked_local()
    closed, analysis, controller = [read(BASE / name) for name in
                                    ['closed.json', 'analysis.json', 'controller/state.json']]
    assert closed['passed'] and analysis['passed']
    assert closed['analysis'] == pin(BASE / 'analysis.json')
    verified_files(BASE, closed['files'])
    assert controller['complete'] and controller['code'] == 0
    assert controller['phase'] == 'collected-and-audited'
    identities = [controller['supervisor']] + [r['child'] for r in controller['stages']]
    for identity in identities:
        try:
            assert psutil.Process(identity['pid']).create_time() != identity['birth']
        except psutil.NoSuchProcess:
            pass
    collection = read(BASE / 'collected/collection.json')
    assert collection['terminal'] and collection['input_error'] is None and collection['code'] == 0
    assert closed['collection'] == pin(BASE / 'collected/collection.json')
    assert analysis['timing_calls'] == 96 and analysis['measured'] == 72 and analysis['warmup'] == 24
    assert analysis['role_labels'] == ROLE_LABELS
    assert analysis['meetings']['passed'] and analysis['meetings']['calls'] == 3
    gate(analysis['reports'])
    campaign = BASE / 'collected/campaign'
    results = [read(campaign / f'timing-{i:02}-{role}-output/result.json')
               for i, role in enumerate(TIMING_ROLES)]
    clocks = []
    for index, (role, result) in enumerate(zip(TIMING_ROLES, results, strict=True)):
        assert len(result['records']) == 16
        for row in result['records']:
            clocks.append(dict(process=index, role=role, **{k: row[k] for k in
                ['name', 'phase', 'start_ticks', 'end_ticks', 'frequency']}))
    assert len(clocks) == 96
    for row in analysis['table']:
        for role in (*ROLES, 'ort'):
            times, means = [], []
            for assigned, result in zip(TIMING_ROLES, results, strict=True):
                if assigned != role:
                    continue
                values = [Fraction(r['end_ticks'] - r['start_ticks'], r['frequency'])
                          for r in result['records'] if r['name'] == row['name'] and r['phase'] == 'measured']
                assert len(values) == 3 and all(v > 0 for v in values)
                times.extend(values)
                means.append(float(sum(values) / 3))
            assert len(times) == 6
            assert float(sum(times) / 6) == row[role]['seconds']
            assert means == [r['mean'] for r in row[role]['processes']]
            assert float(min(times)) == row[role]['minimum'] and float(max(times)) == row[role]['maximum']
    performance = evaluate(analysis['table'])
    assert performance == analysis['performance']
    body = dict(closure=pin(BASE / 'closed.json'), controller=pin(BASE / 'controller/state.json'),
                terminal_local_identities=identities, raw_clocks=clocks, **analysis)
    observations.write_text(json.dumps(body, indent=2) + '\n', encoding='utf8')
    table = '\n'.join(f"| {r['name']} | {r['production']['seconds']:.6f} | {r['portable']['seconds']:.6f} | {r['ort']['seconds']:.6f} | {r['ratios_to_ort']['portable']:.4f} |" for r in analysis['table'])
    controls = '\n'.join(f"| {r['name']} | {r['role']} | {r['process_ratio']:.6f} | {r['limit']:.2f} | {r['passed']} |" for r in performance['controls'])
    gains = '\n'.join(f"| {r['name']} | {r['ratio']:.6f} | {r['limit']:.2f} | {r['passed']} |" for r in performance['gains'])
    full = next(r for r in analysis['table'] if r['audio_seconds'] == 30)
    verdict = 'ADMITTED' if performance['admitted'] else 'NOT ADMITTED'
    report.write_text(f'''# Direct-output Pyannote versus production and Microsoft ORT on AMD

The direct-output candidate is **{verdict} by the original application gates**.
Numerical, application and resource qualification passes. Every timing sample
is retained. These descriptive means do not establish a confidence interval.

| Workload | Production s | Direct-output candidate s | Microsoft ORT s | Candidate / ORT |
|---|---:|---:|---:|---:|
{table}

Candidate / production on the complete dialogue is
{full['portable']['seconds'] / full['production']['seconds']:.6f}.
Candidate / ORT is {full['ratios_to_ort']['portable']:.6f}; the <=1.05 parity
target remains separate from the improvement gate.

AMD EPYC 9V74, CPU2, .NET 10.0.8, SDK 10.0.204 and Microsoft ORT 1.29.0.
Timers include features, neural inference, clustering and owned result creation.
Loading, file access and external validation are excluded. ORT uses one
intra/inter-op thread, sequential execution, full optimization and no spinning.

Six fresh processes run production, candidate, ORT, ORT, candidate, production.
Each performs one warmup and three measured passes over the full dialogue and
three ten-second crops: 96 requests, 24 warmups and 72 measured calls. Each mean
contains six measured requests across two processes. All 96 raw integer clocks,
all process means and every qualification report are retained in
[observations](observations-20260922.json). The candidate is named `portable`
inside the unchanged timing protocol and machine-readable observations.

## Candidate and qualification

Production is the selected Core `e9c87932` / Data `85d166b5`. The candidate is
Core `19b9007d` / Data `cb6f86b0`. It writes eligible grouped convolution rows
directly to final output, with the qualified two-column helper. Remaining rows
use the existing kernel and bias/copy sequence. Scratch allocation, tensor views,
public APIs and packing limits are unchanged. This is the
[Windows-qualified normal composition](../direct-composition/results-20260922.md),
with [complete graph and shared-model checks](../direct-models/results-20260922.md).
This report itself does not integrate the candidate into root source.

Normal Linux builds match all 3,113 Core and 697 Data methods and public
declarations. Complete suites pass {analysis['operator_tests']['backend']['passed']:,} backend tests
with {len(analysis['operator_tests']['backend']['skipped'])} skips and {analysis['operator_tests']['tensors']['passed']:,} tensor tests.
Both actual-caller modes pass all 400 cases: every non-NaN bit, guard and input
is preserved, with NaN classification preserved and payload differences recorded.

Fresh qualification covers both managed roles: 36 Pyannote graph arrays /
5,834,214 values, 32 public requests and 1,568 Parakeet arrays / 6,180,988 values.
All original native gates pass. Candidate Pyannote graphs and public results
match production exactly. Fresh ORT conformance adds four requests. The separate
Parakeet arithmetic candidate and its pending speed selection are not included.

The candidate completes both 600-second meetings and 30-second recovery.
Ordinary and exclusive native speaker timelines match exactly; maximum meeting
centroid error is {analysis['meetings']['maximum_centroid_error']:.9g}.
Maximum timing-call centroid error is {analysis['maximum_centroid_error']:.9g}.
All {analysis['resource_samples']:,} resource samples pass; peak owned RSS is
{analysis['peak_rss']:,} bytes. Local and remote owners are terminal before collection/reporting.

## Original gates

Repeatability controls passed: **{performance['controls_passed']}**.
Speed thresholds passed: **{performance['speed_threshold_passed']}**.
Every role's process-mean max/min must be <=1.10 full and <=1.20 each crop.
Candidate full mean must be <=0.97 of production; each crop <=1.05.
Every gate is mandatory. There is no sample exclusion or unchanged timing retry.

| Workload | Role | Process max/min | Limit | Pass |
|---|---|---:|---:|---|
{controls}

| Workload | Candidate / production | Limit | Pass |
|---|---:|---:|---|
{gains}

## Evidence

Artifact: artifacts/pyannote-direct-amd-execution-v2-20260922.
Closure: {pin(BASE / 'closed.json')['sha256']}.
Analysis: {pin(BASE / 'analysis.json')['sha256']}.
Collection: {pin(BASE / 'collected/collection.json')['sha256']}.
The [protocol](../direct-amd-v2/README.md) was frozen before deployment.
The first campaign stopped at an empty final compiler sample before model or
timing work; its failure remains closed at
5bfd1e51e4aa6788e9470be5b8be8d26db424af8cbbeb25fbb0ef8364ae9f8a4.
This successor verifies actual root/descendant termination before recording an
empty final snapshot separately. Candidate bytes and all live-resource,
numerical and speed gates are unchanged.
''', encoding='utf8')
    print(json.dumps(dict(report=str(report.relative_to(ROOT)), admitted=performance['admitted'],
        controls=performance['controls_passed'], speed=performance['speed_threshold_passed'],
        full_seconds={r: full[r]['seconds'] for r in (*ROLES, 'ort')}, closure=pin(BASE / 'closed.json'))))


if __name__ == '__main__':
    main()
