"""Report every completed comparison sample and the original selection gates."""
from fractions import Fraction
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'tests/pyannote/portable-amd-integration'))
from transport import BASE, PREPARED, SITE, checked_local
from candidate_protocol import pin, read, verified_files, ROLES, TIMING_ROLES
from admission import evaluate, ROLE_LABELS
sys.path.insert(0, str(SITE))
import psutil


def main():
    report, observations = TOOLS / 'results-20260922.md', TOOLS / 'observations-20260922.json'
    assert not report.exists() and not observations.exists()
    checked_local()
    closed, analysis, controller = [read(BASE / name) for name in ['closed.json', 'analysis.json', 'controller/state.json']]
    assert closed['passed'] and analysis['passed'] and closed['analysis'] == pin(BASE / 'analysis.json')
    verified_files(BASE, closed['files'])
    assert controller['complete'] and controller['code'] == 0 and controller['phase'] == 'collected-and-audited'
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
    campaign = BASE / 'collected/campaign'
    results = [read(campaign / f'timing-{i:02}-{role}-output/result.json') for i, role in enumerate(TIMING_ROLES)]
    for row in analysis['table']:
        for role in (*ROLES, 'ort'):
            times, means = [], []
            for assigned, result in zip(TIMING_ROLES, results, strict=True):
                if assigned != role:
                    continue
                values = [Fraction(r['end_ticks'] - r['start_ticks'], r['frequency']) for r in result['records']
                    if r['name'] == row['name'] and r['phase'] == 'measured']
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
        terminal_local_identities=identities, **analysis)
    observations.write_text(json.dumps(body, indent=2) + '\n', encoding='utf8')
    table = '\n'.join(f"| {r['name']} | {r['production']['seconds']:.6f} | {r['portable']['seconds']:.6f} | {r['ort']['seconds']:.6f} | {r['ratios_to_ort']['portable']:.4f} |" for r in analysis['table'])
    controls = '\n'.join(f"| {r['name']} | {r['role']} | {r['process_ratio']:.6f} | {r['limit']:.2f} | {r['passed']} |" for r in performance['controls'])
    gains = '\n'.join(f"| {r['name']} | {r['ratio']:.6f} | {r['limit']:.2f} | {r['passed']} |" for r in performance['gains'])
    full = next(r for r in analysis['table'] if r['audio_seconds'] == 30)
    verdict = 'ADMITTED by the prospective integration gates' if performance['admitted'] else 'NOT ADMITTED by the prospective integration gates'
    report.write_text(f'''# Portable pyannote versus root production and Microsoft ORT on AMD

The portable candidate is **{verdict}**. Numerical, application and resource
qualification passes. Every timing sample is retained; these descriptive means
do not establish calibrated parity or a confidence interval.

| Workload | Root production s | Portable candidate s | Microsoft ORT s | Portable / ORT |
|---|---:|---:|---:|---:|
{table}

Portable uses {100 * (1 - full['portable']['seconds'] / full['production']['seconds']):.3f}% less time than
root production on the complete dialogue. Its ratio to ORT is
{full['ratios_to_ort']['portable']:.6f}; the target of <=1.05 remains separate.

AMD EPYC9V74, CPU2, .NET10.0.8, SDK10.0.204 and Microsoft ORT1.29.0.
Timers include features, neural inference, clustering and owned result creation.
Loading, file access and external validation are excluded. ORT uses one
intra/inter-op thread, sequential execution, full optimization and no spinning.

Six fresh processes run production, portable, ORT, ORT, portable, production.
Each performs one warmup and three measured passes over four fixtures:96requests,
24warmups and72measured calls. Each mean retains six measured requests across
two processes. Raw integer clocks, minimum/maximum and all process means are
retained in [observations](observations-20260922.json).

## Exact candidates and scope

Production is Core d1f86a73 / Data e7fe1668. Portable is Core e9c87932 / Data
85d166b5, the [normal portable source and package](../portable-integration-tests/results-20260922.md).
It includes spatial panels and contiguous copies, ordered LSTM projections and
checked optional storage, pooled outputs, request-scoped contexts, portable
three-row convolution products and sparse mel filters. Public APIs are unchanged.
Neither the unselected AVX-512-first composition nor deferred views is included.

This prospective trial answers a new integration question. It does not change
the [preceding combined-only selection failure](../combined-amd-results/results-20260922.md)
or retrospectively select that trial's portable control. Root integration is
a subsequent, separately qualified build; this report itself changes no product.

## Qualification and resources

Normal Linux portable builds match all3,108Core and697Data methods of the
measured runtime. Complete suites pass {analysis['operator_tests']['backend']['passed']:,} backend tests
with {len(analysis['operator_tests']['backend']['skipped'])} skips and342tensor tests. The required shared
AVX-512 row-kernel test executes; the unselected convolution path is absent.

Exact-runtime closed AMD correctness evidence is reused for each managed role:
18pyannote arrays/16public requests and784Parakeet arrays/3,090,494values, with
all original native and ownership gates passing. All source/runtime/consumer
identities and the original closed reports are verified before reuse; prior
timing results are not selection inputs. Fresh native conformance adds four
pyannote requests. Historical Windows Parakeet discrepancies remain separate.

The portable runtime newly completes both600smeetings and30srecovery on AMD.
Ordinary and exclusive native speaker timelines match exactly. Maximum meeting
centroid error is {analysis['meetings']['maximum_centroid_error']:.9g}; maximum timing-call
centroid error is {analysis['maximum_centroid_error']:.9g}. All{analysis['resource_samples']:,}resource
samples pass, with peak owned RSS{analysis['peak_rss']:,}bytes. Actual local and
remote owners are terminal before collection/reporting.

## Prospective gates

Repeatability controls passed: **{performance['controls_passed']}**.
Speed thresholds passed: **{performance['speed_threshold_passed']}**.
Every role's process-mean max/min must be <=1.10full and<=1.20each crop.
Portable full mean must be <=0.97of root production; each crop<=1.05.
No sample exclusion or unchanged retry is permitted.

| Workload | Role | Process max/min | Limit | Pass |
|---|---|---:|---:|---|
{controls}

| Workload | Portable / production | Limit | Pass |
|---|---:|---:|---|
{gains}

## Evidence

Artifact: artifacts/pyannote-portable-amd-execution-20260922.
Closure: {pin(BASE / 'closed.json')['sha256']}.
Analysis: {pin(BASE / 'analysis.json')['sha256']}.
Collection: {pin(BASE / 'collected/collection.json')['sha256']}.
The [protocol](../portable-amd-integration/README.md) was frozen before deployment.
''', encoding='utf8')
    print(json.dumps(dict(report=str(report.relative_to(ROOT)), admitted=performance['admitted'],
        controls=performance['controls_passed'], speed=performance['speed_threshold_passed'],
        full_seconds={r: full[r]['seconds'] for r in (*ROLES, 'ort')}, closure=pin(BASE / 'closed.json'))))


if __name__ == '__main__':
    main()
