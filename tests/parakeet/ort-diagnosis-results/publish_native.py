"""Publish observed ORT phases and operators, with instrumentation costs visible."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tests/parakeet/ort-diagnosis-amd'))
from run import BASE, pin, read


def main():
    closed = read(BASE/'closed.json')
    assert closed['passed'] and closed['analysis'] == pin(BASE/'analysis.json')
    value = read(BASE/'analysis.json'); folder = BASE/'collected'
    receipt = read(folder/'collection.json')
    assert pin(folder/'collection.json') == closed['collection']
    for name, wanted in receipt['files'].items():
        assert pin(folder/name) == wanted
    out = Path(__file__).resolve().parent
    paths = [out/name for name in ['ort-phases-20260924.md', 'ort-nodes-20260924.csv', 'ort-observations-20260924.json']]
    assert not any(p.exists() for p in paths)
    control, profile = value['phases']['control'], value['phases']['profile']
    lines = ['# Parakeet: the installed ORT workload', '',
        'This is diagnostic attribution of the original twenty-clip native application.',
        'It preserves ORT 1.29.0, original graph outputs, all decoder decisions,',
        'thread settings, input checks and held-output checks. No product is changed.', '',
        '| Complete corpus | Boundary observer, no ORT profiler | With ORT profiler |',
        '|---|---:|---:|']
    for phase in ['frontend', 'encoder', 'decoder']:
        lines.append(f"| {phase} | {control['corpus_phase_seconds'][phase]:.6f}s | {profile['corpus_phase_seconds'][phase]:.6f}s |")
    lines.extend([f"| Outside graph calls | {control['corpus_remainder_seconds']:.6f}s | {profile['corpus_remainder_seconds']:.6f}s |",
        f"| **Complete request** | **{control['corpus_seconds']:.6f}s** | **{profile['corpus_seconds']:.6f}s** |", '',
        f"The immediately preceding uninstrumented native comparison is {value['original_native_seconds']:.6f}s.",
        f"Boundary observation/control is {value['control_over_original']:.6f} times that comparison;",
        f"ORT profiling is {value['profile_over_control']:.6f} times the observed control.",
        'These are separate sequential processes, so the differences include ordinary',
        'process variation. No overhead is subtracted. Diagnostic clocks do not',
        'replace the release benchmark or establish an optimization speedup.', '',
        'Each process completes one warmup and three measured passes: 80 requests,',
        '4,960 graph calls. Measured phases are summed over all twenty clips and',
        'divided by three passes. Every session run is reconciled in order to an',
        'observed graph call. Warmup remains separate. Nested operator intervals',
        'are subtracted from parents so exclusive time is counted once.', '',
        '## Observed operator cost', '',
        'Times below come from the profiled process and include profiler effects.',
        'Operator labels identify ORT graph kernels, not their internal native leaves.', '',
        '| Graph | Operator | Exclusive seconds per corpus | Calls per corpus |',
        '|---|---|---:|---:|'])
    totals = defaultdict(lambda: dict(us=0, calls=0))
    node_rows = []
    for graph, report in value['profiles'].items():
        for row in report['node_clocks']:
            total = totals[(graph, row['op'])]
            total['us'] += row['exclusive_us']; total['calls'] += row['calls']
            node_rows.append(dict(graph=graph, **row, corpus_exclusive_seconds=row['exclusive_us']/3e6))
    for (graph, op), row in sorted(totals.items(), key=lambda item: -item[1]['us']):
        lines.append(f"| {graph} | {op} | {row['us']/3e6:.6f} | {row['calls']/3:g} |")
    lines += ['', '## Identity and limits', '', value['build_info'], '',
        'Actual loaded native libraries are checked against the application payload.',
        'The provider is CPUExecutionProvider, one intra/inter-op thread, sequential',
        'execution, all graph optimizations, spinning disabled; target CPU2 and',
        'monitor CPU0. Every request retains its original numerical/result checks.', '',
        'The native operator profile does not establish exact MLAS dispatch or weight',
        'preparation policy. [The separate graph and native instruction investigation](ort-kernels-20260924.md)',
        'now establishes those observations and their limits. Matched Lokad phase/operator',
        'attribution is required before reporting excess time by component.', '',
        '[Every operator](ort-nodes-20260924.csv) and',
        '[phases, profile accounting and raw-evidence identities](ort-observations-20260924.json).',
        'All runtime shapes remain in the raw artifact analysis.json; the tracked',
        'report avoids duplicating that16MB observation.', '',
        'Raw artifact: `artifacts/parakeet-ort-diagnosis-amd-20260924`.',
        'Closure SHA256: '+pin(BASE/'closed.json')['sha256']+'.', '']
    with paths[1].open('x', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(node_rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(node_rows)
    with paths[2].open('x', encoding='utf8') as stream:
        compact = {k: v for k, v in value.items() if k != 'profiles'}
        compact['graphs'] = {graph: dict(session_calls=p['session_calls'], nodes=len(p['nodes']),
            shape_records=len(p['shapes']), phase_totals=p['phase_totals'], categories=p['categories'])
            for graph, p in value['profiles'].items()}
        json.dump(dict(closure=pin(BASE/'closed.json'), raw_analysis=pin(BASE/'analysis.json'),
                       raw_artifact='artifacts/parakeet-ort-diagnosis-amd-20260924', **compact), stream, indent=2)
    paths[0].write_text('\n'.join(lines), encoding='utf8')
    print(json.dumps(dict(report=pin(paths[0]), phases=control['corpus_phase_seconds'],
                         profile_over_control=value['profile_over_control'])))


if __name__ == '__main__':
    main()
