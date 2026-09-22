"""Publish both reconciled diagnostics and all timestamps without selecting speed."""
import json
from pathlib import Path
from protocol import pin, read, save

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'artifacts/pyannote-blocked-spatial-phases-amd-20260922'
OUTPUT = Path(__file__).resolve().parent


def main():
    assert not (OUTPUT/'results-20260922.md').exists() and not (OUTPUT/'observations-20260922.json').exists()
    proof = read(BASE/'closed.json'); assert proof['passed'] and proof['diagnostic_only']
    for name, wanted in proof['files'].items(): assert pin(BASE/name) == wanted, name
    analysis = read(BASE/'analysis.json'); assert analysis['complete'] and analysis['passed'] and analysis['calls'] == 864
    reports = {name: read(BASE/'collected'/name/'result.json') for name in ['profile-a', 'profile-b']}
    census = read(ROOT/'artifacts/pyannote-blocked-spatial-census-20260922/census.json')
    save(OUTPUT/'observations-20260922.json', dict(closure=pin(BASE/'closed.json'), analysis=analysis,
        captures=reports, qualification=read(BASE/'collected/qualify-512/result.json'), forms=census['forms']))
    processes = analysis['processes']; first, second = [processes[n] for n in ['profile-a', 'profile-b']]
    lines = ['# AMD complete spatial convolution phase attribution', '',
        'Both instrumented captures reconcile every phase to the complete owned',
        'result interval. All 864 calls pass: 768 eligible and 96 fallback calls.',
        'Every output matches selected production exactly and every operand stays',
        'unchanged. The generated arithmetic and original model qualifier are',
        'byte-identical to the preceding actual-model qualification.', '',
        'These durations include instrumentation. They attribute the failed prototype;',
        'they do not reopen its speed gate, measure complete diarization, or compare',
        'against Microsoft ORT. The selected implementation remains unchanged.', '',
        '| Phase | Capture A seconds | A share | Capture B seconds | B share |',
        '|---|---:|---:|---:|---:|']
    for phase in first['total']['phases']:
        a = first['total']['phases'][phase]; b = second['total']['phases'][phase]
        lines.append(f'| {phase.replace("_", " ")} | {a["seconds"]:.9f} | {a["share"]*100:.3f}% | {b["seconds"]:.9f} | {b["share"]*100:.3f}% |')
    lines += [f'| Total | {first["total"]["total"]["seconds"]:.9f} | 100% | {second["total"]["total"]["seconds"]:.9f} | 100% |', '',
        'Each total sums all 108 calls for three crops, averaged over three measured',
        'passes within that process. One complete warmup pass per process is retained',
        'separately. Original multiplicities are already included, without another',
        'census multiplier. Diagnostic total max/min across processes is',
        f'{analysis["diagnostic_process_max_min"]["seconds"]:.9f}. No speed-selection threshold is applied.', '',
        '| Form | Calls per crop | Eligible | A input / arithmetic / output share | B input / arithmetic / output share |',
        '|---:|---:|---|---|---|']
    for form in census['forms']:
        key = str(form['index']); a = first['forms'][key]['phases']; b = second['forms'][key]['phases']
        def shares(row): return ' / '.join(f'{row[k]["share"]*100:.2f}%' for k in ['input_layout', 'arithmetic', 'output_epilogue'])
        lines.append(f'| {key} | {form["multiplicity"]} | {form["eligible"]} | {shares(a)} | {shares(b)} |')
    lines += ['', 'The output phase includes layout conversion, bias, optional residual addition',
        'and activation. Input includes padding clear and layout conversion. Validation',
        'includes overlap/extents checks and finite-operand scans. Allocation includes',
        'the result and scratch rentals; pool return and wrapper costs stay explicit.',
        'Method-boundary gaps and all timestamp overhead remain in the inclusive sum.',
        'Fallbacks use their complete public interval and cannot reuse stale phase data.', '',
        'All nine nested timestamps are monotonic and the independent auditor derives',
        'phases from their intervals, rather than trusting reported phase totals.',
        'Every record retains sequence/thread identity, warmup status and output hash.',
        'Eight tests reject missing/duplicate calls, nonmonotonic clocks, wrong clock',
        'frequency, stale fallback data, one-tick losses and incorrect outputs.', '',
        'Before capture, the exact executable passes all 108 actual-model cases and',
        '119,823,360 values on AMD. Maximum native scaled error remains',
        '`3.814697265625e-6`, below `1e-4`. CPU2 is inherited before CLR startup;',
        'monitor CPU0, .NET 10.0.8, selected Core `1279b4b6`, no runtime overrides.',
        f'All {analysis["samples"]} resource samples pass, peak target RSS {analysis["peak_rss"]:,} bytes.',
        'Both captures, qualification and supervisor are actually terminal with exit code zero.', '',
        f'Independent closure: `{pin(BASE/"closed.json")["sha256"]}`.',
        f'Analysis SHA-256: `{pin(BASE/"analysis.json")["sha256"]}`.',
        f'Raw timestamps SHA-256: `{pin(OUTPUT/"observations-20260922.json")["sha256"]}`.',
        '[All observations](observations-20260922.json); [prospective protocol](README.md).',
        'The failed component screen and its original gates remain preserved separately.']
    (OUTPUT/'results-20260922.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    save(BASE/'published-report.json', dict(report=pin(OUTPUT/'results-20260922.md'), raw=pin(OUTPUT/'observations-20260922.json'), generator=pin(Path(__file__))))
    print(json.dumps(dict(report=pin(OUTPUT/'results-20260922.md'), raw=pin(OUTPUT/'observations-20260922.json'))))


if __name__ == '__main__': main()
