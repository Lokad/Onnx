"""Check all retained outputs twice; report numerical failures without hiding them."""
import collections
import datetime
import json
import math
from protocol import *


def check_metric(actual, expected):
    whole = metric(actual, expected)
    chunks = metric(actual, expected, True)
    for key in ['max_scaled', 'failed_values', 'values']:
        assert whole[key] == chunks[key], (key, whole, chunks)
    assert math.isclose(whole['squared_error'], chunks['squared_error'], rel_tol=1e-12, abs_tol=1e-18)
    return whole


def main():
    assert not (BASE/'closed.json').exists() and not (BASE/'analysis.json').exists()
    spec = read(BASE/'manifest.json'); state = read(BASE/'processes.json')
    assert spec['protocol'] == PROTOCOL and state['manifest'] == pin(BASE/'manifest.json')
    assert state['complete'] and state['code'] == 0 and len(state['runs']) == len(spec['jobs']) == 8
    assert [r['original'] for r in spec['requests']] == SELECTED and spec['limits'] == LIMITS
    identities = [state['supervisor']]+[r['worker'] for r in state['runs']]
    assert all(absent(identity) for identity in identities)
    for name, expected in spec['files'].items():
        assert pin(ROOT/name) == expected, name
    for name, expected in spec['numerical_files'].items():
        assert pin(name) == expected, name
    summaries = []; result_pins = {}; sample_count = 0; peak = 0; scalar_count = 0
    for job, run in zip(spec['jobs'], state['runs'], strict=True):
        assert run['job'] == job and run['complete'] and run['code'] == 0 and run['samples'] > 0
        folder = BASE/'outputs'/job['id']; result = read(folder/'result.json')
        assert result['complete'] and result['job'] == job and result['manifest'] == pin(BASE/'manifest.json')
        request = spec['requests'][job['request']]
        assert result['input_unchanged'] and result['input_sha256'] == request['input']['raw_sha256']
        runtime = result['runtime']; assert {k:runtime[k] for k in ['pid', 'birth']} == run['worker']
        assert runtime['affinity'] == [2] and runtime['blas_threads'] == 1 and runtime['native_loaded'] is False
        for name, expected in runtime['loaded'].items():
            assert expected == spec['numerical_files'][name], name
        assert len(result['records']) == spec['nodes']
        assert dict(collections.Counter(r['op'] for r in result['records'])) == spec['census']
        assert [r['index'] for r in result['records']] == list(range(spec['nodes']))
        assert all(r['dtype'] == 'float32' for r in result['records'])
        expected_dots = spec['census']['MatMul'] if job['mode'] == 'wide-matmul' else 0
        assert len(result['scalar_checks']) == expected_dots
        for row in result['scalar_checks']:
            assert len(row['checks']) == 3
            for check in row['checks']:
                error = abs(check['actual']-check['expected'])/max(1., abs(check['expected']))
                assert error == check['max_scaled'] and error <= spec['scalar_limit']
                scalar_count += 1
        samples = [json.loads(line) for line in (BASE/'process'/job['id']/'samples.jsonl').read_text().splitlines()]
        assert len(samples) == run['samples']
        assert max(s['rss'] for s in samples) == run['peak_rss']
        for sample in samples:
            assert {k:sample[k] for k in ['pid', 'birth']} == run['worker'] and sample['affinity'] == [2]
            assert 0 <= sample['seconds'] < LIMITS['seconds'] and sample['rss'] < LIMITS['rss']
            assert sample['available'] >= LIMITS['available'] and sample['disk'] >= LIMITS['disk']
        sample_count += len(samples); peak = max(peak, run['peak_rss'])
        assert len(result['outputs']) == len(spec['outputs']) == 41
        boundaries = []; pins = []
        for index, (row, desc) in enumerate(zip(result['outputs'], spec['outputs'], strict=True)):
            assert row['index'] == index and row['name'] == desc['name'] and row['shape'] == desc['shape']
            path = folder/row['file']; assert pin(path) == row['pin']
            actual = np.fromfile(path, dtype='<f4').reshape(desc['shape']); pins.append(row['pin'])
            expected_bytes = math.prod(desc['shape'])*4; assert path.stat().st_size == expected_bytes
            refs = {}
            for engine in ['numpy', 'ort']:
                reference = request['references'][engine][index]
                assert reference['name'] == desc['name'] and reference['shape'] == desc['shape']
                value = np.fromfile(ROOT/reference['file'], dtype='<f8').reshape(desc['shape'])
                refs[engine] = check_metric(actual, value)
            boundaries.append(dict(index=index, name=desc['name'], references=refs))
        direct = {kind:check_metric(actual, source(desc)) for kind,desc in request['baselines'].items()}
        result_pins[job['id']] = pins
        summaries.append(dict(job=job, name=request['name'], boundaries=boundaries, original_fp32_final=direct,
                              diagnostic_seconds=result['seconds'], scalar_coordinates=sum(len(r['checks']) for r in result['scalar_checks'])))
    for mode in MODES:
        assert result_pins['00-'+mode] == result_pins['03-'+mode], mode
    screening = []
    for request in spec['requests']:
        selected = {row['job']['mode']:row for row in summaries if row['job']['request'] == request['index']}
        for engine in ['numpy', 'ort']:
            baseline = selected['float32']['boundaries'][-1]['references'][engine]
            wide = selected['wide-matmul']['boundaries'][-1]['references'][engine]
            screening.append(dict(request=request['index'], reference=engine,
                maximum_ratio=wide['max_scaled']/baseline['max_scaled'],
                passed=wide['max_scaled'] <= spec['useful_maximum_ratio']*baseline['max_scaled'] and wide['failed_values'] <= baseline['failed_values']))
    analysis = dict(protocol=PROTOCOL, structural_passed=True, useful_signal=all(s['passed'] for s in screening),
        jobs=summaries, screening=screening, arrays=328, comparisons=8*(41*2+2), metric_checks=8*(41*2+2)*4,
        scalar_coordinates=scalar_count, samples=sample_count, peak_rss=peak, births=identities,
        manifest=pin(BASE/'manifest.json'), source_revision=spec['source_revision'], utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    rows = ['# Whisper matrix-accumulation precision diagnostic — 2026-09-21', '',
        'The experiment changes only MatMul accumulation in an unfused float32 NumPy/SciPy encoder. '
        'The wider variant uses float64 products and sums, then rounds every MatMul output to float32. '
        'All other node semantics and float32 output boundaries remain identical. This is not a Lokad kernel '
        'or an ORT FP32 execution; transfer to the managed fused graph requires a separate implementation.', '',
        '| Request | Mode | Final max vs NumPy FP64 | Failed values | Final max vs ORT FP64 | Failed values |',
        '|---|---|---:|---:|---:|---:|']
    for row in summaries:
        a, b = (row['boundaries'][-1]['references'][e] for e in ['numpy', 'ort'])
        rows.append(f"| {row['job']['request']}: {row['name']} | {row['job']['mode']} | {a['max_scaled']:.9g} | {a['failed_values']:,} | {b['max_scaled']:.9g} | {b['failed_values']:,} |")
    signal = analysis['useful_signal']
    rows += ['', '**The prospective mechanism screen '+('passes' if signal else 'fails')+'**: '
        'it requires at least a halving of final maximum error on every selected case against both references, '
        'with no increase in failed-value counts. '+('This nominates a managed-kernel prototype; it does not qualify one.' if signal else
        'This experiment does not justify implementing a wide managed matrix kernel on its own.'), '',
        'All scaled comparisons use `abs(actual-reference)/max(1,abs(reference))`, with failures strictly above '
        '`1e-4`. The complete observations retain all 41 boundaries, both references and final comparisons '
        'against the original managed and native FP32 outputs. Historical failures and product defaults are unchanged.', '',
        'The fixed inputs are three previously selected natural recordings, managed features only, followed '
        'by the first recording again. These include prior large-discrepancy cases; they are not held-out '
        'accuracy evidence. Both variants repeat every captured array bit for bit. All padded frames are included.', '',
        f"All eight workers and the supervisor are terminal. All 328 output arrays, {sample_count:,} resource samples, "
        f"{analysis['metric_checks']:,} independent metric checks and {scalar_count:,} scalar dot coordinates pass structural verification. "
        f"Peak sampled RSS is {peak:,} bytes. CPU 2 and one numerical-library thread are verified. "
        'Observed Python durations are diagnostics, not performance comparisons.', '',
        '[Prior whole-layer-20 evidence](../layer20-reference/results-20260921.md) motivates the experiment. '
        '[Complete observations](observations-20260921.json) retain all comparisons. '
        f"Artifact: `artifacts/whisper-matmul-precision-20260921`; frozen source `{spec['source_revision']}`. "
        f"Manifest SHA256 `{analysis['manifest']['sha256']}`.", '']
    report = Path(__file__).with_name('results-20260921.md'); observations = report.with_name('observations-20260921.json')
    assert not report.exists() and not observations.exists()
    write(BASE/'analysis.json', analysis); write(observations, analysis)
    with report.open('x', encoding='utf8') as stream:
        stream.write('\n'.join(rows))
    files = {p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}
    write(BASE/'closed.json', dict(structural_passed=True, useful_signal=signal, files=files, births=identities,
        reports={rel(report):pin(report), rel(observations):pin(observations)}))
    print(json.dumps(dict(closed=True, useful_signal=signal, arrays=328, receipt=pin(BASE/'closed.json'))))


if __name__ == '__main__':
    main()
