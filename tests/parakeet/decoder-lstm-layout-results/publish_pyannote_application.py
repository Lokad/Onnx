"""Publish complete Pyannote admission, preserving every clock and failed control."""
import json
from pathlib import Path
import sys
from publish_application import csvfile, pin, read

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-decoder-lstm-layout-pyannote-app-amd-20260927'
sys.path.insert(0, str(ROOT / 'tests/parakeet/pad-current-pyannote-app-amd'))
from admission import evaluate
from checks import timing_table
from protocol import JOBS, TIMING_ROLES


def clocks_and_setups(collected):
    clocks = []
    setups = []
    reports = []
    for index, role in enumerate(TIMING_ROLES):
        name = f'timing-{index:02}-{role}'
        value = read(collected / name / 'output/result.json')
        reports.append(value)
        setups.append(dict(process=name, seconds=value['setup_seconds']))
        for ordinal, row in enumerate(value['records']):
            clocks.append(dict(process=name, role=role, index=ordinal,
                              **{k: row[k] for k in ['name', 'phase', 'start_ticks',
                                                     'end_ticks', 'frequency', 'seconds']}))
    assert len(clocks) == 96 and sum(r['phase'] == 'measured' for r in clocks) == 72
    return reports, clocks, setups


def render(value, closure):
    performance = value['performance']
    assert evaluate(value['table']) == performance
    controls = sum(row['passed'] for row in performance['controls'])
    gates = sum(row['passed'] for row in performance['gates'])
    assert len(performance['controls']) == 12 and len(performance['gates']) == 4
    lines = ['# Prepared LSTM layout: complete Pyannote comparison', '',
             '**All application regression gates pass.**' if performance['admitted']
             else '**Application regression gates do not all pass.**', '',
             '| Fixture | Current seconds | Candidate seconds | Microsoft ORT seconds | Candidate / ORT | Candidate / current |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in value['table']:
        lines.append(f"| {row['name']} | {row['selected']['seconds']:.6f} | "
                     f"{row['candidate']['seconds']:.6f} | {row['ort']['seconds']:.6f} | "
                     f"{row['ratios_to_ort']['candidate']:.6f} | {row['ratios_to_selected']['candidate']:.6f} |")
    lines += ['', f'Repeatability: {controls}/12 controls pass; full-dialogue process max/min <= 1.10, crops <= 1.20.',
              f'Regression: {gates}/4 gates pass; candidate/current <= 1.05.']
    failed = [f"- Repeatability: {r['name']} / {r['role']}: {r['process_ratio']:.6f} > {r['limit']:.2f}."
              for r in performance['controls'] if not r['passed']]
    failed += [f"- Regression: {r['name']}: {r['ratio']:.6f} > {r['limit']:.2f}."
               for r in performance['gates'] if not r['passed']]
    if failed:
        lines += ['', 'Failed checks:', '', *failed]
    lines += ['', 'AMD EPYC 9V74 CPU 2, .NET 10.0.8 and ORT 1.29.0 CPUExecutionProvider.',
              'The clock includes frontend, segmentation, embeddings, clustering and owned',
              'public results. Setup and validation are separate. Six fresh processes run',
              'current, candidate, ORT, ORT, candidate, current. Every process uses one',
              'warmup and three measured passes over each of the four fixtures.',
              'All 96 requests, including 24 warmups and 72 measurements, remain in the',
              'published clocks. The unchanged scorer retains every clock with equal',
              'process weights and exact clock fractions. No profiling or timing correction.', '',
              'Four fresh native public requests pass the original conformance bounds.',
              'Candidate timed public results equal the current root exactly, including',
              'centroids. Both 600-second meetings and the 30-second recovery preserve',
              'the original complete reference results and the native/portable bounds.',
              'Input immutability and held-output ownership checks remain intact.',
              'These checks establish compatibility; they do not measure human-label accuracy.', '',
              f"All nine workers and their supervisor are terminal. {sum(r['samples'] for r in value['resources']):,}",
              f"resource observations pass; peak owned RSS is {max(r['peak_rss'] for r in value['resources']):,} bytes.", '',
              'This comparison evaluates the unchanged LSTM layout selected on Parakeet.',
              'Its [component screen](timing-20260927.md) remains rejected because',
              '100/168 repeatability controls failed. The',
              '[runtime diagnosis](../decoder-lstm-runtime-results/README.md) finds',
              'JIT activity within measured calls, but does not correct or rescore',
              'the original clocks or resolve all prepared-path variation.',
              'Actual root/package qualification still precedes release promotion and the',
              'BENCHMARK.md update. A failed admission remains failed; no unchanged retry.', '',
              '[Every clock](pyannote-application-clocks-20260927.csv),',
              '[all timing setup intervals](pyannote-application-setups-20260927.csv),',
              '[complete results, controls and identities](pyannote-application-20260927.json).', '',
              f"Closure: `{closure['sha256']}`.",
              f'Raw evidence: `{BASE.relative_to(ROOT).as_posix()}`.']
    return '\n'.join(lines) + '\n'


def main():
    proof = read(BASE / 'closed.json')
    value = read(BASE / 'analysis.json')
    payload = read(BASE / 'payload.json')
    assert proof['passed'] and value['passed'] and proof['analysis'] == pin(BASE / 'analysis.json')
    for name, wanted in proof['files'].items():
        assert pin(BASE / name) == wanted, name
    assert value['identities'] == payload['identities']
    parakeet = read(OUT / 'application-20260927.json')['identities']
    assert value['identities']['selected'] == parakeet['current']
    assert value['identities']['candidate'] == parakeet['candidate']
    assert value['reference_provenance_verified']
    assert (value['native_public_requests'], value['meeting_requests'], value['timing_requests'],
            value['warmup'], value['measured']) == (4, 3, 96, 24, 72)
    assert proof['admitted'] == value['performance']['admitted']
    collected = BASE / 'collected'
    state = read(collected / 'identity.json')
    receipt = read(collected / 'collection.json')
    assert state['complete'] and state['code'] == 0 and receipt['terminal'] and receipt['code'] == 0
    assert [r['name'] for r in state['runs']] == JOBS
    assert all(r['complete'] and r['code'] == 0 for r in state['runs'])
    assert value['results']['meetings-run']['complete_selected_results_exact']
    assert all(r['passed'] for r in value['results'].values())
    for name in ['timing-01-candidate', 'timing-04-candidate']:
        assert value['results'][name]['cross_product']['full_results_exact']
    reports, clocks, setups = clocks_and_setups(collected)
    assert timing_table(reports, read(collected / 'manifests/selected-pyannote.json')) == value['table']
    closure = pin(BASE / 'closed.json')
    prose = render(value, closure)
    paths = {suffix: OUT / ('pyannote-application-' + suffix) for suffix in
             ['20260927.md', '20260927.json', 'clocks-20260927.csv', 'setups-20260927.csv']}
    assert not any(path.exists() for path in paths.values()), 'Preserve an existing publication'
    with paths['20260927.json'].open('x', encoding='utf8') as stream:
        json.dump(dict(closure=closure, **value), stream, indent=2, allow_nan=False)
    csvfile(paths['clocks-20260927.csv'], clocks)
    csvfile(paths['setups-20260927.csv'], setups)
    with paths['20260927.md'].open('x', encoding='utf8') as stream:
        stream.write(prose)
    print(json.dumps(dict(passed=True, admitted=proof['admitted'], closure=closure,
                          clocks=len(clocks), setups=len(setups))))


if __name__ == '__main__':
    main()
