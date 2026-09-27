"""Publish the closed LSTM layout graph comparison, retaining every clock and verdict."""
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
from publish_application import pin, read, csvfile

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-decoder-lstm-layout-graphs-amd-20260927'
TOOLS = ROOT/'tests/parakeet/pad-current-graphs-amd'
sys.path.insert(0, str(TOOLS))
from protocol import CASES, ORDER
loader = importlib.util.spec_from_file_location('pad_graph_statistics', TOOLS/'statistics.py')
statistics = importlib.util.module_from_spec(loader)
loader.loader.exec_module(statistics)


def main():
    proof = read(BASE/'closed.json')
    value = read(BASE/'analysis.json')
    payload = read(BASE/'payload.json')
    assert proof['passed'] and value['passed']
    assert proof['files']['analysis.json'] == pin(BASE/'analysis.json')
    for name, wanted in proof['files'].items():
        assert pin(BASE/name) == wanted, name
    assert value['products'] == payload['products'] and not value['root_product_changed']
    assert value['products']['current']['Lokad.Onnx.dll']['sha256'] == '0d224bcff591563816d64c3a3cc51f7b7cd38f5a1d3c523b05c699a9b82b6b97'
    assert value['products']['candidate']['Lokad.Onnx.dll']['sha256'] == 'ad97b4ad632b3306ea3a39d14549ed98540bc8c2e85953a68d89b453d3ed2fdc'
    assert [r['key'] for r in value['performance']] == CASES
    assert proof['admitted'] == all(r['qualified'] for r in value['performance'])
    assert proof['all_controls_passed'] == all(c['passed'] for r in value['performance'] for c in r['controls'])
    assert (value['clocks'], value['measured'], len(value['setups'])) == (73512, 8640, 72)
    for name in ['consumer', 'e5_consumer', 'short_consumer']:
        assert value[name]['passed'] and value[name]['implementation_flags_equal']
        assert value[name]['branches_locals_exceptions_equal'] and not value[name]['product_changed']
    clocks = (BASE/'clocks.csv').read_text(encoding='utf8')
    rows = list(csv.DictReader(io.StringIO(clocks)))
    assert len(rows) == 73512 and sum(r['warmup'] == 'False' for r in rows) == 8640
    state = read(BASE/'collected/identity.json')
    receipt = read(BASE/'collected/collection.json')
    assert state['complete'] and state['code'] == 0 and receipt['terminal'] and receipt['code'] == 0
    setups = []
    for row in state['runs']:
        result = read(BASE/'collected'/row['name']/'output/result.json')
        setups.append(dict(process=row['name'], seconds=result['setup_seconds']))
    assert setups == value['setups']
    lines = ['# Prepared LSTM layout: complete graph comparison', '',
        '**All graph regression gates pass.**' if proof['admitted'] else '**Graph regression gates do not all pass.**', '',
        '| Case | Current seconds | Candidate seconds | Microsoft ORT seconds | Candidate / ORT | Candidate / current |',
        '|---|---:|---:|---:|---:|---:|']
    for row in value['performance']:
        reports = {role: read(BASE/'collected'/f"timing-{row['key']}-{role}"/'output/result.json') for role in ORDER}
        assert {k: v for k, v in row.items() if k != 'key'} == statistics.summarize(reports)
        lines.append(f"| {row['key']} | {row['current']:.6f} | {row['candidate']:.6f} | {row['ort']:.6f} | {row['ratio']:.6f} | {row['candidate_over_current']:.6f} |")
    controls = sum(c['passed'] for row in value['performance'] for c in row['controls'])
    gates = sum(row['regression_passed'] for row in value['performance'])
    samples = sum(row['samples'] for row in value['resources'])
    peak = max(row['peak_rss'] for row in value['resources'])
    lines += ['', 'AMD EPYC 9V74 CPU 2, .NET 10.0.8 and ORT 1.29.0 CPUExecutionProvider.',
        'Lokad times Reset, Execute and owned output arrays; ORT times session.run.',
        'Setup, input creation and validation are separate. Native execution uses',
        'one intra/inter-op thread, sequential execution and all graph optimizations.', '',
        'All 24 numerical jobs precede the 48 timed jobs. Each case runs current,',
        'candidate, ORT, ORT, candidate, current in six fresh processes. Eight-token',
        'e5 uses 6,000 warmups, thirty-token e5 uses 1,200, and the other six cases',
        'use 600. Every timed process has 180 measurements. All 73,512 calls,',
        '8,640 measurements and 72 setup intervals are retained.',
        'The unchanged scorer uses exact clock fractions and equal process weights.',
        'No clock is trimmed or corrected.', '',
        f'Repeatability: {controls}/24 controls pass (process max/min <= 1.10).',
        f'Regression: {gates}/8 gates pass (candidate/current <= 1.05).']
    failures = []
    for row in value['performance']:
        for control in row['controls']:
            if not control['passed']:
                failures.append(dict(case=row['key'], kind='repeatability', **control))
        if not row['regression_passed']:
            failures.append(dict(case=row['key'], kind='regression',
                                 candidate_over_current=row['candidate_over_current'], limit=1.05))
    if failures:
        lines += ['', 'Failed checks (also retained in the complete JSON):', '']
        lines += ['- ' + json.dumps(row, sort_keys=True, allow_nan=False) for row in failures]
    lines += ['', 'Candidate outputs equal current-root bytes; both managed products satisfy',
        'the fresh ORT scaled-error bound of 1e-4, shapes and finiteness checks.',
        'Input immutability and independently owned held outputs remain checked.',
        'The three previously qualified consumers are reused without compilation',
        'or implementation-flag changes. No inference overlaps another workload.', '',
        f'All owners are terminal. {samples:,} resource observations pass; peak owned RSS is {peak:,} bytes.', '',
        'This comparison qualifies the one fixed LSTM layout change across supported',
        'graphs. It does not attribute timing changes to a particular node or prove',
        'that every graph executes the changed path.', '',
        'The [component screen](timing-20260927.md) remains rejected because',
        '100/168 repeatability controls failed. The separate',
        '[runtime diagnosis](../decoder-lstm-runtime-results/README.md) finds JIT',
        'activity within measured calls; its trace shifts the slow pass and cannot',
        'correct the original clocks. Prepared-path variation remains unresolved.',
        'The admitted [complete Parakeet comparison](application-20260927.md)',
        'independently establishes application benefit for this unchanged candidate.',
        'This graph verdict alone does not promote the product or BENCHMARK.md.',
        'Pyannote application and actual root/package qualification still apply.', '',
        '[Every clock](graphs-clocks-20260927.csv), [all setups](graphs-setups-20260927.csv),',
        '[complete controls, identities and resources](graphs-20260927.json).', '',
        'Closure: `' + pin(BASE/'closed.json')['sha256'] + '`.']
    paths = [OUT/name for name in ['graphs-20260927.md', 'graphs-20260927.json',
                                  'graphs-clocks-20260927.csv', 'graphs-setups-20260927.csv']]
    assert not any(p.exists() for p in paths), 'Preserve the existing publication'
    with paths[0].open('x', encoding='utf8') as stream:
        stream.write('\n'.join(lines) + '\n')
    with paths[1].open('x', encoding='utf8') as stream:
        json.dump(dict(closure=pin(BASE/'closed.json'), admitted=proof['admitted'], **value), stream, indent=2, allow_nan=False)
    with paths[2].open('x', encoding='utf8', newline='') as stream:
        stream.write(clocks)
    csvfile(paths[3], setups)
    print(json.dumps(dict(passed=True, admitted=proof['admitted'], closure=pin(BASE/'closed.json'),
                         cases=len(CASES), controls=controls, gates=gates, clocks=len(rows))))


if __name__ == '__main__':
    main()
