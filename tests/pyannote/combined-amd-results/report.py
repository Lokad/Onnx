"""Report every completed comparison sample and the original selection gates."""
from fractions import Fraction
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'tests/pyannote/combined-amd-v2'))
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
    assert analysis['timing_calls'] == 128 and analysis['measured'] == 96 and analysis['warmup'] == 32
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
    table = '\n'.join(f"| {r['name']} | {r['production']['seconds']:.6f} | {r['portable']['seconds']:.6f} | {r['rows']['seconds']:.6f} | {r['ort']['seconds']:.6f} | {r['ratios_to_ort']['rows']:.4f} |" for r in analysis['table'])
    controls = '\n'.join(f"| {r['name']} | {r['role']} | {r['process_ratio']:.6f} | {r['limit']:.2f} | {r['passed']} |" for r in performance['controls'])
    gains = '\n'.join(f"| {r['name']} | {r['control']} | {r['ratio']:.6f} | {r['limit']:.2f} | {r['passed']} |" for r in performance['gains'])
    full = next(r for r in analysis['table'] if r['audio_seconds'] == 30)
    verdict = 'ADMITTED by the original descriptive selection gates' if performance['admitted'] else 'NOT ADMITTED by the original descriptive selection gates'
    report.write_text(f'''# Actual combined pyannote candidate versus Microsoft ORT on AMD

The complete numerical, application and resource audit passes. The combined
candidate is **{verdict}**. Every timing sample is retained; no calibrated
confidence interval or parity claim is inferred from these means.

| Workload | Previous rows s | Current portable s | Combined s | Microsoft ORT s | Combined / ORT |
|---|---:|---:|---:|---:|---:|
{table}

Full dialogue combined/previous rows is {full['rows']['seconds'] / full['production']['seconds']:.6f},
and combined/current portable is {full['rows']['seconds'] / full['portable']['seconds']:.6f}.

AMD EPYC 9V74, CPU2, .NET10.0.8, SDK10.0.204 and ORT1.29.0. Timers include
features, neural inference, decoding/clustering and owned result creation.
Loading, file access and external validation are excluded. ORT uses one
intra/inter-op thread, sequential execution and its original optimization settings.

The eight processes run previous rows, portable, combined, ORT, ORT, combined,
portable, previous rows. Each performs one warmup and three measured passes
over four fixtures: 128 requests, 32 warmups and 96 measured calls. Each displayed
mean has six measured requests across two processes. Raw integer clocks drive
the table, with every minimum, maximum and process mean retained in
[observations](observations-20260922.json).

## Exact candidates and scope

Protocol key production means the previous isolated AVX-512 rows candidate,
Core294/Datae7. It does not mean root production. Portable is the current normal
Coree9c/Data85 build. Rows is the actual combined Coree369/Data2b51 build:
AVX-512 convolution first, newer portable fallback, output pooling, request
contexts, sparse mel filters and checked LSTM storage. The separate deferred-view
experiment is excluded. The complete source and package qualification is in
[combined-avx512](../combined-avx512/results-20260922.md).

The original campaign stopped before inference in the portable graph checker
because its Data identity literal still named the older assembly. The correction
changes exactly that literal in each affected consumer. Fourteen independently
closed successful stages were reused; the failed stage and every timing stage
were excluded from reuse. Products, assertions, timing consumers and gates are
unchanged. [Recovery evidence](../combined-amd-review/recovery-20260922.md)
preserves both the refusal and exact compiled-method proof.

## Qualification and resources

AMD passes {analysis['operator_tests']['backend']['passed']:,} backend tests
with {len(analysis['operator_tests']['backend']['skipped'])} skips, 342 tensor
tests and all three mandatory AVX-512 execution tests. The normal Linux build
matches all 3,111 Core and 697 Data methods of the qualified runtime.
All three roles pass 18 pyannote arrays and 16 public requests each, all 784
Parakeet trajectory arrays / 3,090,494 values each, and the existing native and
ownership checks. Fresh native conformance adds four pyannote calls. These AMD
Parakeet passes do not erase the historical Windows numerical failures.

Both ten-minute meetings and short recovery preserve native ordinary/exclusive
timelines and satisfy the centroid bound. Maximum meeting centroid error is
{analysis['meetings']['maximum_centroid_error']:.9g}; maximum timing-call centroid
error is {analysis['maximum_centroid_error']:.9g}. All
{analysis['resource_samples']:,} resource samples pass; peak owned RSS is
{analysis['peak_rss']:,} bytes. All actual local and remote identities are terminal.

## Original gates

Repeatability controls passed: **{performance['controls_passed']}**.
Speed thresholds passed: **{performance['speed_threshold_passed']}**.
Full process-mean max/min must be <=1.10; every crop must be <=1.20 for every role.
Combined full latency must be <=0.97 of both managed controls; each crop <=1.05.
No sample is excluded and no unsuccessful unchanged protocol is retried.

| Workload | Protocol role | Process max/min | Limit | Pass |
|---|---|---:|---:|---|
{controls}

| Workload | Managed control key | Combined / control | Limit | Pass |
|---|---|---:|---:|---|
{gains}

## Evidence

Artifact: artifacts/pyannote-combined-amd-execution-v2-20260922.
Closure: {pin(BASE / 'closed.json')['sha256']}.
Analysis: {pin(BASE / 'analysis.json')['sha256']}.
Collection: {pin(BASE / 'collected/collection.json')['sha256']}.
This report performs no new inference and does not itself apply a product patch.
''', encoding='utf8')
    print(json.dumps(dict(report=str(report.relative_to(ROOT)), performance=performance,
        full=full, closure=pin(BASE / 'closed.json'))))


if __name__ == '__main__':
    main()
