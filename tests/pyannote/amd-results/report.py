"""Publish the completed AMD campaign without modifying its executed tools."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
CAMPAIGN_TOOLS = ROOT / 'tests/pyannote/amd-candidates'
sys.path.insert(0, str(CAMPAIGN_TOOLS))
from candidate_protocol import ROLES, TIMING_ROLES, pin, read, verified_files
from transport import BASE, PREPARED, SITE, checked_local
sys.path.insert(0, str(SITE))
import psutil


def terminal(identity):
    try:
        assert psutil.Process(identity['pid']).create_time() != identity['birth'], identity
    except psutil.NoSuchProcess:
        pass


def main():
    report, observations = TOOLS / 'results-20260922.md', TOOLS / 'observations-20260922.json'
    assert not report.exists() and not observations.exists()
    prepared, bundle, execution = checked_local()
    closure, analysis, controller = (read(BASE / p) for p in ['closed.json', 'analysis.json', 'controller/state.json'])
    assert closure['passed'] and analysis['passed'] and closure['analysis'] == pin(BASE / 'analysis.json')
    assert controller['complete'] and controller['code'] == 0 and controller['phase'] == 'collected-and-audited'
    verified_files(BASE, closure['files'])
    identities = [controller['supervisor']] + [row['child'] for row in controller['stages']]
    for identity in identities:
        terminal(identity)
    receipt = read(BASE / 'collected/collection.json')
    assert receipt['terminal'] and not receipt['input_error'] and closure['collection'] == pin(BASE / 'collected/collection.json')
    campaign = BASE / 'collected/campaign'
    state = read(campaign / 'identity.json')
    assert state['complete'] and state['code'] == 0
    assert analysis['timing_calls'] == 128 and analysis['measured'] == 96 and analysis['warmup'] == 32
    results = [read(campaign / f'timing-{i:02}-{role}-output/result.json') for i, role in enumerate(TIMING_ROLES)]
    # Recompute the displayed table from the complete integer-clock records.
    spec = importlib.util.spec_from_file_location('audited_amd_table', CAMPAIGN_TOOLS / 'audit_results.py')
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    table = auditor.timing_table(results, read(PREPARED / 'payload/manifests/production-pyannote.json'))
    assert table == analysis['table'] and len(table) == 4
    variation = []
    for row in table:
        for role in (*ROLES, 'ort'):
            process_means = [p['mean'] for p in row[role]['processes']]
            assert len(process_means) == 2
            variation.append(dict(name=row['name'], role=role, process_means=process_means,
                process_max_min=max(process_means) / min(process_means)))
    runtimes = {role: {name: pin(PREPARED / 'payload/runtimes' / role / name)
        for name in ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll', 'AudioBenchmark.dll']} for role in ROLES}
    body = dict(closure=pin(BASE / 'closed.json'), controller=pin(BASE / 'controller/state.json'),
        controller_identities=identities, runtimes=runtimes, variation=variation, **analysis)
    observations.write_text(json.dumps(body, indent=2) + '\n', encoding='utf8')
    table_text = '\n'.join(f"| {row['name']} | {row['production']['seconds']:.6f} | {row['portable']['seconds']:.6f} | {row['rows']['seconds']:.6f} | {row['ort']['seconds']:.6f} | {row['ratios_to_ort']['portable']:.4f} | {row['ratios_to_ort']['rows']:.4f} |" for row in table)
    variation_text = '\n'.join(f"| {r['name']} | {r['role']} | {r['process_means'][0]:.6f} | {r['process_means'][1]:.6f} | {r['process_max_min']:.4f} |" for r in variation)
    identities_text = '\n'.join(f"- {role}: Core `{value['Lokad.Onnx.dll']['sha256']}`, Data `{value['Lokad.Onnx.Data.dll']['sha256']}`." for role, value in runtimes.items())
    backend, tensors = analysis['operator_tests']['backend'], analysis['operator_tests']['tensors']
    report.write_text(f'''# AMD pyannote: production, portable, AVX-512 and Microsoft ORT

The complete campaign passes its independent numerical, application and
resource audit. All **128 timing requests** are retained: **32 warmups** and
**96 measured calls**. This is a descriptive matched comparison on the target
VM; it does not establish calibrated parity or promote production automatically.

| Workload | Production s | Portable s | AVX-512 rows s | Microsoft ORT s | Portable / ORT | Rows / ORT |
|---|---:|---:|---:|---:|---:|---:|
{table_text}

AMD EPYC9V74,logicalCPU2,.NET10.0.8,SDK10.0.204,MicrosoftORT1.29.0. All workers
use their original complete public application timers: frontend processing,
neural execution,clustering and owned outputs are included; loading,file access
and external validation are excluded. ORT uses one intra/inter-op thread and
sequential execution. No runtime override or sample exclusion is introduced.

Eight fresh processes execute production,portable,rows,ORT,ORT,rows,portable,
production. Each performs one complete warmup and three measured passes over
all four fixtures. Six measured calls contribute to each role/fixture mean.
All individual samples,minima,maxima and process means are retained in
[observations](observations-20260922.json) and the immutable collected evidence.

## What these candidates contain

Production is the original CoreD1 build. Portable Core469 combines spatial
convolution panels,contiguous copies and ordered LSTM projections. Core294 adds
the convolution-specific AVX-512 row-sharing path adapted from voice cdeae16.
All three use identical Datae7 and the same application consumer. They do not
contain the later output pooling,request contexts,portable three-row composition,
sparse mel,LSTM storage guard or deferred-view trial. The final composition
therefore requires its own complete qualification and fresh ORT comparison.

{identities_text}

## Correctness and resources

The normal AMD build passes **{backend['passed']:,} backend tests** with
**{len(backend['skipped'])} skips**, and **{tensors['passed']} tensor tests** with
**{len(tensors['skipped'])} skips**. All three specifically required AVX-512
hardware tests execute successfully. The instruction comparison verifies the
built Core/Data against the pinned row-sharing runtime.

Every role passes all18pyannote graph arrays and16publicrequests. A further
four native public conformance requests pass. Each also passes all784Parakeet
trajectory arrays/3,090,494values at the original1e-4scaled numerical bound,
including its original public/rejection/recovery checks. These AMD native passes
do not erase the separate Windows numerical failures. All128timing calls also
pass the original native/input/held-output checks; maximum centroid error is
{analysis['maximum_centroid_error']:.9g}.

All **{analysis['resource_samples']:,} resource samples** pass; peak aggregate
worker RSS is **{analysis['peak_rss']:,}bytes**. Original bounds are fourhours
overall,onehour perworker,12GiBRSS,1GiBminimumavailable/temporaryfree and2GiB
campaignfiles. Preflight requires12GiBavailable and3GiBtemporaryfree. Source,
builds and outputs use bounded /dev/shm storage. The evidence collection confirms
the actual remote processes are terminal; the local controller and its children
are also terminal before this report is written.

## Observed process variation

| Workload | Role | First process mean s | Second process mean s | Max / min |
|---|---|---:|---:|---:|
{variation_text}

These ratios expose variation; they are not a newly introduced acceptance test.
The frozen primary protocol specifies descriptive timing after complete
numerical/resource qualification. No retrospective threshold,confidence interval
or removal of a slower process is used. Candidate choice for the next integrated
experiment must consider every fixture and both processes. The newer portable
and AVX-512 predicates overlap,so composition order must be explicit.

## Evidence

Artifact: `artifacts/pyannote-amd-execution-v4-20260921`.
Closure: `{pin(BASE / 'closed.json')['sha256']}` ({pin(BASE / 'closed.json')['bytes']:,}bytes).
Analysis: `{pin(BASE / 'analysis.json')['sha256']}`.
Collection: `{pin(BASE / 'collected/collection.json')['sha256']}`.
This reporter reads the closed campaign; it does not rerun inference or modify
the frozen execution tools. Use C:/Python313/python.exe -X utf8 -B with report.py
in this directory after the existing collection/audit controller actually exits.
Existing report outputs are refused.
''', encoding='utf8')
    print(json.dumps(dict(report=str(report.relative_to(ROOT)), table=table, variation=variation,
        closure=pin(BASE / 'closed.json'))))


if __name__ == '__main__':
    main()
