"""Publish one closed parent or release comparison, including every clock."""
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PRODUCTS = {
    'parent': '40260aef7fd93c5153601ec104a87843a2c017a2720fd64c9e24e3460d455749',
    'release': 'f95a13c58354bf07f3b7926b72903c18b1a560a56673297cb9fe001d3541b592',
}
CANDIDATE = 'e07a45189b348fe55ce76300415c6c35ba6a2bc0d224f1fc13b0b92c303bccba'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size,
                    sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def csvfile(path, rows):
    with path.open('x', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(mode):
    assert mode in PRODUCTS
    base = ROOT/f'artifacts/parakeet-owned-batch-isolation-{mode}-app-amd-20260925'
    proof, value, payload = [read(base/name) for name in ['closed.json', 'analysis.json', 'payload.json']]
    assert proof['passed'] and value['passed'] and proof['analysis'] == pin(base/'analysis.json')
    for name, wanted in proof['files'].items():
        assert pin(base/name) == wanted, name
    assert value['identities'] == payload['identities']
    identities = payload['identities']
    assert identities['current']['Lokad.Onnx.dll']['sha256'] == PRODUCTS[mode]
    assert identities['candidate']['Lokad.Onnx.dll']['sha256'] == CANDIDATE
    assert identities['candidate']['Lokad.Onnx.Data.dll']['sha256'] == '01e9e7842f5e9861de3d6dc737db947c8a38f1a07038b403d5482ec676e810f1'
    expected_data = identities['candidate']['Lokad.Onnx.Data.dll']['sha256'] if mode == 'parent' else 'a893952f583f680ad9dcf677a32b9393541814396a35c6a4eb18a1e7325cbae1'
    assert identities['current']['Lokad.Onnx.Data.dll']['sha256'] == expected_data
    assert not payload['release_admitted'] and payload['failed_graph_cases'] == []
    assert not value['root_product_changed']
    assert (value['timing_requests'], value['warmup'], value['measured']) == (480, 120, 360)
    performance = value['performance']
    assert proof['admitted'] == performance['admitted']
    assert len(performance['controls']) == 63 and len(performance['gates']) == 21
    corpus, = [row for row in value['table'] if row['is_corpus']]
    assert corpus['audio_seconds'] == 213.265 and len(value['table']) == 21
    gain = 1 - corpus['candidate']['seconds']/corpus['current']['seconds']
    controls = sum(row['passed'] for row in performance['controls'])
    gates = sum(row['passed'] for row in performance['gates'])
    corpus_gate, = [row for row in performance['gates'] if row['name'].startswith('corpus-')]
    assert corpus_gate['limit'] == (1.05 if mode == 'parent' else 0.97)
    rows = [dict(name=row['name'], audio_seconds=row['audio_seconds'],
                 baseline_seconds=row['current']['seconds'],
                 candidate_seconds=row['candidate']['seconds'], ort_seconds=row['ort']['seconds'],
                 candidate_over_ort=row['ratios_to_ort']['candidate'],
                 candidate_over_baseline=row['candidate']['seconds']/row['current']['seconds'])
            for row in value['table']]
    clocks = []
    for index, role in enumerate(['current', 'candidate', 'ort', 'ort', 'candidate', 'current']):
        name = f'timing-{index:02}-{role}'
        result = read(base/'collected'/name/'output/result.json')
        for request, row in enumerate(result['records']):
            clocks.append(dict(process=name, role=role, index=request, **{key: row[key] for key in
                ['name', 'phase', 'start_ticks', 'end_ticks', 'frequency', 'seconds']}))
    assert len(clocks) == 480 and sum(row['phase'] == 'measured' for row in clocks) == 360
    stem = f'{mode}-application-20260925'
    paths = {suffix: OUT/(stem+suffix) for suffix in ['.md', '.json', '.csv', '-clocks.csv']}
    assert not any(path.exists() for path in paths.values()), 'Preserve existing publication'
    baseline = 'direct-depthwise parent' if mode == 'parent' else 'qualified release'
    requirement = ('at most 5% corpus regression against the direct-depthwise parent'
                   if mode == 'parent' else 'at least 3% corpus improvement over the qualified release')
    failed = [f"- Repeatability: {row['name']} / {row['role']}, {row['process_ratio']:.6f} > {row['limit']:.2f}."
              for row in performance['controls'] if not row['passed']]
    failed += [f"- Performance: {row['name']}, candidate/baseline {row['candidate_over_current']:.6f} > {row['limit']:.2f}."
               for row in performance['gates'] if not row['passed']]
    failure_text = ('\n\nFailed checks:\n\n'+'\n'.join(failed)) if failed else ''
    provenance = ('This is a retention check after moving the packed-weight dispatch decision. '
                  'It does not require another optimization gain against the successful depthwise parent. '
                  'The separate direct-release comparison retains its original 3% improvement requirement.'
                  if mode == 'parent' else
                  'This compares the actual release and candidate directly, with their respective Core and Data binaries. '
                  'Retained outputs for these exact products also match byte for byte across 1,568 tensor pairs '
                  'and 40 complete public result pairs in both instruction modes. No historical times are pooled.')
    prose = f'''# Dispatch relocation: complete Parakeet comparison against the {baseline}

**The candidate {'passes' if performance['admitted'] else 'fails'} the prospective application gates.**
Complete transcription latency is {100*abs(gain):.3f}% {'lower' if gain >= 0 else 'higher'} than the {baseline}.
Repeatability: {controls}/63 checks. Performance: {gates}/21 gates.
Candidate/ORT is **{corpus['ratios_to_ort']['candidate']:.6f}**; the separate
1.05 parity target is {'met' if performance['parity_target_met'] else 'not met'}.{failure_text}

| Twenty-clip corpus (213.265 s audio) | Seconds | Relative to ORT |
| --- | ---: | ---: |
| {baseline.capitalize()} | {corpus['current']['seconds']:.6f} | {corpus['ratios_to_ort']['current']:.6f} |
| Dispatch-relocation candidate | {corpus['candidate']['seconds']:.6f} | {corpus['ratios_to_ort']['candidate']:.6f} |
| Microsoft ORT 1.29.0 | {corpus['ort']['seconds']:.6f} | 1.000000 |

Six fresh processes run baseline, candidate, ORT, ORT, candidate, baseline on
AMD EPYC 9V74 CPU 2. Each uses one warmup and three measured passes per clip:
480 complete requests, 120 warmups and 360 measurements. The existing public
consumer, numerical validators and exact-clock scorer are unchanged. No clock
is trimmed, and no profiling or runtime override is enabled.

The prospective limits require {requirement},
no clip more than 5% slower, corpus process max/min <=1.10 and per-clip max/min
<=1.20 for all three engines. All workers are terminal/code0. All
{sum(row['samples'] for row in value['resources']):,} resource samples pass;
peak owned RSS is {max(row['peak_rss'] for row in value['resources']):,} bytes.

{provenance}

[Full Parakeet correctness](models-20260925.md) and
[combined graph qualification](../../benchmarks/e5-steady-short-results/qualified-graphs-20260925.md)
are separate completed proofs. Shared/Pyannote and actual root/package
qualification remain required before promotion. Root source and BENCHMARK.md
are unchanged. A failed verdict does not permit an unchanged retry.

[Every case]({stem}.csv), [all 480 clocks]({stem}-clocks.csv), and
[all controls, product identities and resources]({stem}.json).
Closure: `{pin(base/'closed.json')['sha256']}`.
Raw evidence: `{base.relative_to(ROOT).as_posix()}`.
'''
    with paths['.json'].open('x', encoding='utf8') as stream:
        json.dump(dict(closure=pin(base/'closed.json'), **value,
                       comparison=mode, release_admitted=False), stream, indent=2, allow_nan=False)
        stream.write('\n')
    csvfile(paths['.csv'], rows)
    csvfile(paths['-clocks.csv'], clocks)
    with paths['.md'].open('x', encoding='utf8') as stream:
        stream.write(prose)
    print(json.dumps(dict(comparison=mode, admitted=performance['admitted'], gain=gain,
        candidate_over_ort=corpus['ratios_to_ort']['candidate'], controls=controls, gates=gates,
        release_admitted=False, report=pin(paths['.md']))))


if __name__ == '__main__':
    assert len(sys.argv) == 2
    main(sys.argv[1])
