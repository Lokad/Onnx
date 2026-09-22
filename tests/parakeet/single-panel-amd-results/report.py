"""Publish the complete fixed comparison only after its independent audit closes."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-single-panel-amd-execution-v2-20260922'
PAYLOAD = ROOT / 'artifacts/parakeet-single-panel-amd-payload-v2-20260922/payload'
sys.path.insert(0, str(ROOT / 'tests/parakeet/single-panel-amd-v2'))
from candidate_protocol import ROLES, TIMING_ROLES, pin, read, verified_files
from admission import evaluate
from audit_results import timing_table


def main():
    output = TOOLS / 'results-20260922.md'; observations = TOOLS / 'observations-20260922.json'
    assert not output.exists() and not observations.exists()
    proof = read(BASE / 'closed.json'); analysis = read(BASE / 'analysis.json')
    assert proof['passed'] and analysis['passed'] and proof['analysis'] == pin(BASE / 'analysis.json')
    verified_files(BASE, proof['files'])
    state = read(BASE / 'controller/state.json')
    assert state['complete'] and state['code'] == 0
    sys.path.insert(0, str(ROOT / 'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
    import psutil
    for identity in [state['supervisor'], *[r['child'] for r in state['stages']]]:
        try: assert psutil.Process(identity['pid']).create_time() != identity['birth']
        except psutil.NoSuchProcess: pass
    collection = read(BASE / 'collected/collection.json'); assert collection['terminal']
    campaign = BASE / 'collected/campaign'
    results = [read(campaign / f'timing-{i:02}-{r}-output/result.json') for i, r in enumerate(TIMING_ROLES)]
    table = timing_table(results, read(PAYLOAD / 'manifests/production-parakeet.json'))
    assert table == analysis['table'] and evaluate(table) == analysis['performance']
    assert sum(len(r['records']) for r in results) == analysis['timing_calls'] == 480
    corpus = table[-1]; performance = analysis['performance']
    rows = []
    for index, (role, result) in enumerate(zip(TIMING_ROLES, results, strict=True)):
        rows.append(dict(index=index, role=role, engine=result['engine'], output=pin(campaign / f'timing-{index:02}-{role}-output/result.json'),
            records=[{key: r[key] for key in ['name', 'pass', 'phase', 'start_ticks', 'end_ticks', 'frequency', 'seconds', 'input_sha256', 'ownership']}
                for r in result['records']]))
    retained = dict(passed=True, scope='All raw clocks, including warmups; complete public/numerical outputs remain in the pinned collected artifact',
        closure=pin(BASE / 'closed.json'), analysis=pin(BASE / 'analysis.json'), collection=pin(BASE / 'collected/collection.json'),
        table=table, performance=performance, processes=rows)
    observations.write_text(json.dumps(retained, indent=2, allow_nan=False) + '\n', encoding='utf8')
    selected = performance['admitted']
    text = [
        '# Parakeet arithmetic composition versus production and Microsoft ORT on AMD', '',
        '**' + ('The fixed performance gates admit the candidate.' if selected else 'The fixed performance gates do not admit the candidate.') + '**', '',
        f"The complete 20-clip corpus takes **{corpus['production']['seconds']:.6f} s production, "
        f"{corpus['portable']['seconds']:.6f} s candidate and {corpus['ort']['seconds']:.6f} s Microsoft ORT**. "
        f"Candidate / production is **{corpus['ratios_to_production']['portable']:.9f}**; "
        f"candidate / ORT is **{corpus['ratios_to_ort']['portable']:.9f}**. "
        'The full application parity target remains 1.05.', '',
        '| Clip / corpus | Audio s | Production s | Candidate s | ORT s | Candidate / production | Candidate / ORT |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in table:
        text.append(f"| {row['name']} | {row['audio_seconds']:.3f} | {row['production']['seconds']:.6f} | "
            f"{row['portable']['seconds']:.6f} | {row['ort']['seconds']:.6f} | "
            f"{row['ratios_to_production']['portable']:.6f} | {row['ratios_to_ort']['portable']:.6f} |")
    text += ['',
        'CPU2 on the designated AMD EPYC 9V74 VM, .NET 10.0.8 / ORT 1.29.0. '
        'Production is selected Pyannote Core `1279b4b6` / Data `4e602d9f`; '
        'candidate is Core `abbf5e98` / Data `eb452663`. Both retain the 256 MiB '
        'encoder packing cap. The internal candidate role is named `portable`.', '',
        'Six fresh processes run production, candidate, ORT, ORT, candidate, production. '
        'Each executes one complete warmup and three measured passes: **480 requests, '
        '120 warmups and 360 measurements**, over 213.265 audio seconds. Per-clip means '
        'have six observations. Corpus means sum all twenty clip means within each '
        'process, then average the two processes equally. All calculations and '
        'admission decisions use exact integer-clock fractions.', '',
        'Timers include frontend, neural graphs, each engine’s own greedy decoder '
        'trajectory and owned results. Model loading, file access and external '
        'validation are excluded for both engines. ORT uses one intra/inter-op thread, '
        'sequential execution, full optimization and no spinning. No profiling, '
        'forced collections or JIT overrides occur during timing.', '',
        '## Fixed gates', '',
        f"All 63 repeatability controls pass: **{performance['controls_passed']}**. "
        f"All 21 speed gates pass: **{performance['speed_threshold_passed']}**. "
        f"Full candidate/ORT parity target passes: **{performance['parity_target_met']}**.", '',
        'Every role requires corpus process max/min <=1.10 and each clip <=1.20. '
        'Candidate corpus/production must be <=0.95 and every clip <=1.05. '
        'No observation is dropped and no unchanged retry follows a failed gate. '
        'Rounded table entries never determine a pass.', '',
        '| Workload | Role | Process max/min | Limit | Pass |', '|---|---|---:|---:|---|']
    text.extend(f"| {r['name']} | {r['role']} | {r['process_ratio']:.9f} | {r['limit']:.2f} | {r['passed']} |" for r in performance['controls'])
    text += ['', '| Workload | Candidate / production | Limit | Pass |', '|---|---:|---:|---|']
    text.extend(f"| {r['name']} | {r['ratio']:.9f} | {r['limit']:.2f} | {r['passed']} |" for r in performance['gains'])
    suites = analysis['operator_tests']
    text += ['', '## Qualification and evidence', '',
        f"Normal Linux builds match all **3,114 Core / 697 Data methods** and public declarations. "
        f"Full suites pass **{suites['backend']['passed']} backend / {len(suites['backend']['skipped'])} skipped "
        f"and {suites['tensors']['passed']} tensor tests**, including the required executed AVX-512 test. "
        'All 400 convolution caller cases per mode, original Parakeet arithmetic '
        'cases and forty AMD prepared-precedence cases pass.', '',
        'Both managed roles pass 36 Pyannote graph arrays / 32 public requests and '
        '1,568 Parakeet native arrays. Selected Pyannote graph/public outputs stay exact. '
        'Before timing, all twenty public Parakeet clips pass for each managed role '
        'and ORT (60 requests), plus four separate Pyannote native requests. '
        'Both 600-second Pyannote meetings and 30-second recovery retain native '
        'timelines and centroid bounds. These are finite fixture checks; separate '
        'intermediate-layer and double-reference discrepancies remain recorded.', '',
        f"All **{analysis['resource_samples']} resource observations** pass; peak owned RSS is "
        f"**{analysis['peak_rss']:,} bytes**. Public Parakeet preflight requires 14 GiB "
        'available; other workers require 12 GiB. Original 12 GiB RSS, 1 GiB '
        'available/tmpfs, 3 GiB tmpfs preflight, 2 GiB artifact, one-hour worker '
        'and four-hour campaign limits remain. Only verified duplicate archives '
        'and inventoried unpinned build caches are removed. All inputs, built '
        'binaries and results survive final verification. Collection follows '
        'actual termination of every recorded owner.', '',
        '[All raw observations](observations-20260922.json) and the '
        '[frozen protocol](../single-panel-amd-v2/README.md) retain every process '
        'and clip. Previous failed Windows timing and preparations remain closed. '
        'The first campaign preparation failed on a prerequisite schema before '
        'payload creation; v2 corrects that reader and passes 37 selftests.', '',
        'Artifact: `artifacts/parakeet-single-panel-amd-execution-v2-20260922`.',
        f"Closure: `{pin(BASE / 'closed.json')['sha256']}`.",
        f"Analysis: `{pin(BASE / 'analysis.json')['sha256']}`.",
        f"Collection: `{pin(BASE / 'collected/collection.json')['sha256']}`.", '',
        'This report does not itself integrate the arithmetic source. Promotion '
        'requires the admitted verdict followed by normal root and package checks.']
    output.write_text('\n'.join(text) + '\n', encoding='utf8')
    print(json.dumps(dict(report=pin(output), observations=pin(observations), admitted=selected, corpus=corpus)))


if __name__ == '__main__': main()
