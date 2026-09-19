"""Publish the complete audited comparison without rerunning any inference."""
from pathlib import Path
import argparse, hashlib, json, statistics


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--artifact', type=Path, required=True)
    p.add_argument('--observations', type=Path, required=True); p.add_argument('--report', type=Path, required=True)
    a = p.parse_args(); base = a.artifact.resolve()
    assert not a.observations.exists() and not a.report.exists()
    value = read(base / 'model-audit.json'); qualification = read(base / 'qualification-audit.json')
    freeze = read(base / 'analysis-freeze.json'); meta = read(base / 'provenance.json')
    assert value['integrity_passed'] and value['phase'] == 'model' and value['measured_calls'] == 5940
    assert qualification['passed'] and qualification['product_codegen_inspected']
    assert value['core_sha256'] == qualification['core_sha256'] == meta['core_sha256']
    for name, digest in freeze['files'].items():
        assert sha(base / 'analysis-source' / name) == sha(Path(__file__).with_name(name)) == digest
    assert value['audit_sha256'] == freeze['files']['audit.py'] and value['evaluate_sha256'] == freeze['files']['evaluate.py']
    assert freeze['qualification_audit_sha256'] == sha(base / 'qualification-audit.json')
    assert sha(base / 'collected-model/analysis-freeze.json') == sha(base / 'analysis-freeze.json')
    assert value['archive_sha256'] == sha(base / 'model-results.tar.gz')
    output = dict(schema=1, source=meta['product'], protocol=meta['protocol'], criteria=meta['criteria'],
                  qualification=qualification, comparison=value, prospective_analysis=freeze,
                  evidence={name: sha(base / name) for name in ('model-audit.json', 'qualification-audit.json', 'codegen-review.json',
                      'analysis-freeze.json', 'model-results.tar.gz', 'qual-results.tar.gz', 'payload.tar.gz')})
    with a.observations.open('x', encoding='utf-8') as stream:
        json.dump(output, stream, indent=2); stream.write('\n')
    passed = value['passed']
    if passed:
        opening = 'The opt-in product passes the predeclared empirical performance criteria and every correctness/resource check. It remains off by default; this is not calibrated confidence or a fresh ORT comparison.'
    elif not value['control_passed']:
        opening = 'The identical controls fail the predeclared stability criteria, so the performance conclusion is inconclusive. Every sample remains. Correctness and ownership qualification pass; the route stays off by default.'
    else:
        opening = 'The product comparison fails at least one predeclared candidate or machine-health criterion. Every sample remains, and the route stays off by default. Correctness and ownership qualification pass.'
    lines = ['# Exact zero-block softmax: complete e5 comparison — 2026-09-19', '', opening, '',
        'Product `087e280`, AMD EPYC 9V74, .NET 10.0.8, SDK 10.0.204, logical CPU 2 before runtime startup. '
        'Actual public `ExecutionOptions.Memory`, normal tiering and GC. Both controls have identical settings; only the candidate sets '
        '`LOKAD_ONNX_SOFTMAX_ZERO_BLOCKS=1`. The [protocol](README.md) was frozen before model observations.', '',
        'Ninety fresh processes cover six balanced visits to all five cases. Each conditions for thirty cumulative Execute seconds '
        'before 33 Execute calls and a separate 33 Reset-plus-Execute calls. All 5,940 measured calls remain, together with '
        f"{sum(w['conditioning_calls'] for w in value['workers']):,} conditioning calls. No outlier removal, forced GC or CPU-time substitution.", '']
    for boundary, title in (('execute', 'Public Execute'), ('request', 'Complete request: Reset plus Execute')):
        lines += ['## ' + title, '', '| Case | Mean controls ms | Candidate ms | Candidate / controls | Control B / A |',
                  '|---|---:|---:|---:|---:|']
        for name, case in value['cases'].items():
            row = case[boundary]; means = row['mean_ms']; control = (means['controlA'] + means['controlB']) / 2
            lines.append(f"| {name} | {control:.4f} | {means['candidate']:.4f} | {row['candidate_ratio']:.6f} | {row['control_ratio']:.6f} |")
        lines += ['', 'Controls are the arithmetic mean of both roles. Each role has six process visits and 198 measured calls per case. '
                  'These descriptive values are not ratios to historical ORT measurements.', '']
    lines += ['## Prospective criteria and process variation', '',
        'For both boundaries: control B/A within 1% in aggregate and 3% in every triplet; padded128 candidate improves at least 1% '
        'in aggregate with every triplet ratio at most 1.02; all other cases at most 2% aggregate regression and 5% per triplet. '
        'Foreign CPU is limited to 2% of machine capacity and steal to 0.5%. The limits were not revised after observation.', '',
        f"Control criteria: **{'pass' if value['control_passed'] else 'fail'}**. Candidate criteria: **{'pass' if value['candidate_passed'] else 'fail'}**. "
        f"Machine health: **{'pass' if all(value['health'].values()) else 'fail'}**. Overall: **{'pass' if passed else 'fail'}**.", '',
        '| Case | Boundary | Control triplet ratio range | Candidate triplet ratio range | Failed criteria |', '|---|---|---:|---:|---|']
    for name, case in value['cases'].items():
        for boundary, row in case.items():
            failed = [k for group in ('control_criteria', 'candidate_criteria') for k, passed_gate in row[group].items() if not passed_gate]
            control = row['control_visit_ratios']; candidate = row['candidate_visit_ratios']
            lines.append(f"| {name} | {boundary} | {min(control):.6f}–{max(control):.6f} | {min(candidate):.6f}–{max(candidate):.6f} | {', '.join(failed) or 'none'} |")
    telemetry = value['telemetry']
    lines += ['', '## Correctness, generated code and resources', '',
        'Before timing, local full suites passed 3,027 backend and 342 tensor tests per setting (93 expected backend ISA skips), '
        'plus fallback configurations, native e5 in three modes and full shared-model replays. AMD then passed 280 contracts per setting '
        'and 106 shared arrays / 1,286,766 values per setting. DINOv3, ResNet50 and GPT-2 outputs are byte-identical across settings.', '',
        f"Every full-model worker preserves actual inputs and held outputs. Complete before/after e5 arrays pass the unchanged native "
        f"scaled-error limit of 1e-4; maximum observed error is `{value['maximum_scaled_error']:.12g}`. All exported e5 outputs are "
        'byte-identical across visits and settings for each case. Native ORT is absent from these managed processes.', '',
        'The separate actual-product full Tier1 capture is 3,744 bytes. Its IG43→40→41, IG66→63→64 and IG90→87→88 branches '
        'return zero and bypass the exponential polynomial for the all-lane predicate. This is generated-code inspection, '
        'not a dynamic branch-count measurement; those worker times are excluded.', '',
        f"Maximum sampled model process-group RSS: {telemetry['maximum_rss']:,} bytes. Maximum observed foreign CPU: "
        f"{telemetry['maximum_foreign_cpu_fraction']:.9%} of total machine capacity; maximum steal: {telemetry['maximum_steal_fraction']:.9%}. "
        'All workers remain below the 6-GiB / 600-second model bounds. Snapshot deltas miss some exited or short-lived work '
        'and cannot establish absence of hypervisor interference.', '',
        '| Case | Role | Execute GC 0/1/2 | Request GC 0/1/2 | Request allocated bytes/call |', '|---|---|---|---|---:|']
    for name in value['cases']:
        for role in ('controlA', 'controlB', 'candidate'):
            workers = [w for w in value['workers'] if w['name'] == name and w['role'] == role]
            gc = {b: '/'.join(str(sum(w[b]['gc'][g] for w in workers)) for g in range(3)) for b in ('execute', 'request')}
            allocation = statistics.mean(w['request']['allocated_bytes_per_call'] for w in workers)
            lines.append(f"| {name} | {role} | {gc['execute']} | {gc['request']} | {allocation:.1f} |")
    lines += ['', 'GC events and their wall time are retained. Allocation samples describe this finite protocol, not an indefinite memory guarantee.', '',
        '## Identities and retained evidence', '',
        f"[All measured samples and visit means]({a.observations.name}) have SHA256 `{sha(a.observations)}`. "
        'The artifact `artifacts/softmax-zero-product-20260919` retains full conditioning, arrays, process telemetry, source, binaries and archives. '
        'Collectors verify terminal supervisor/worker identities before creating immutable receipts.', '',
        f"Core: `{meta['core_sha256']}`. Probe: `{meta['probe_sha256']}`. The product DLLs were built immediately before the implementation "
        'commit from the identical source; their assembly metadata predates that commit. The source archive is an identity record, '
        'not an archive-build claim.', '',
        f"Qualification archive: `{output['evidence']['qual-results.tar.gz']}`. Model archive: `{output['evidence']['model-results.tar.gz']}`. "
        f"Model audit: `{output['evidence']['model-audit.json']}`. Prospective analysis freeze: `{output['evidence']['analysis-freeze.json']}`.", '',
        'The independent auditor/estimator has eight tests, including malformed identity/clock/output evidence, all criterion failures, '
        'balanced ordering, use of every sample and arithmetic control averaging. The successful prior long-batch kernel protocol '
        'and the earlier failed small-batch protocol remain separate observations. This result neither rewrites them nor changes '
        'the existing Microsoft ORT scoreboard or production defaults.', '']
    with a.report.open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines))
    print('Wrote', a.report, 'and all', value['measured_calls'], 'measured calls; overall', value['passed'])


if __name__ == '__main__': main()
