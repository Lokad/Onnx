"""Read-only reconciliation of rejected padding screens and retained CLR events.

No inference, product build, original-auditor execution or admission change.
Run from any directory with Python 3.13; writes one new compact observation file.
"""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
INPUTS = {}


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size,
                    sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def read(path, wanted=None):
    identity = pin(path)
    if wanted is not None:
        assert identity == wanted, str(path)
    INPUTS[path.relative_to(ROOT).as_posix()] = identity
    return json.loads(path.read_text(encoding='utf8'))


def closed(name, digest):
    base = ROOT / 'artifacts' / name
    path = base / 'closed.json'
    assert pin(path)['sha256'] == digest
    proof = read(path)
    assert proof['passed'] and not proof['root_product_changed']
    return base, proof


def rational(value):
    return dict(numerator=value.numerator, denominator=value.denominator,
                value=float(value))


def screen(name, digest):
    base, proof = closed(name, digest)
    assert proof['admitted'] is False
    analysis = read(base / 'analysis.json', proof['files']['analysis.json'])
    assert analysis['admitted'] is False
    processes = []
    means = {}
    for sequence, role in enumerate(('current', 'candidate', 'candidate', 'current')):
        process = f'{role}-screen{sequence}-512'
        name = f'collected/{process}/result.json'
        result = read(base / name, proof['files'][name])
        assert result['passed'] and result['flags'] == {} and result['runtime'] == '10.0.8'
        assert result['sequence'] == sequence and result['role'] == role
        assert result['core_sha256'] == analysis['products'][role]['Lokad.Onnx.dll']['sha256']
        assert len(result['rows']) == 12
        frequency = result['frequency']
        assert type(frequency) is int and frequency > 0
        cases = []
        for index, row in enumerate(result['rows']):
            assert row['index'] == index and row['exact'] and row['inputs'] and row['ownership']
            clocks = row['clocks']
            assert len(clocks) == 780
            for iteration, clock in enumerate(clocks):
                assert clock['iteration'] == iteration
                assert clock['warmup'] == (iteration < 600)
                assert type(clock['ticks']) is int and clock['ticks'] > 0
            measured = Fraction(sum(c['ticks'] for c in clocks[600:]), 180 * frequency)
            means[sequence, index] = measured
            blocks = [Fraction(sum(c['ticks'] for c in clocks[i:i + 60]), 60 * frequency)
                      for i in (600, 660, 720)]
            cases.append(dict(index=index, name=row['name'], shape=row['shape'], pads=row['pads'],
                mean_seconds=rational(measured), measured_blocks_seconds=[rational(b) for b in blocks],
                all_call_seconds=rational(Fraction(sum(c['ticks'] for c in clocks), frequency)),
                setup_seconds=rational(Fraction(row['setupTicks'], frequency))))
        processes.append(dict(process=process, role=role, cases=cases,
            all_call_seconds=sum(c['all_call_seconds']['value'] for c in cases),
            note='Sum of public-call clocks, not process elapsed time; excludes setup and validation.'))
    rows = []
    for index, original in enumerate(analysis['rows']):
        selected = (means[0, index] + means[3, index]) / 2
        candidate = (means[1, index] + means[2, index]) / 2
        ratio = candidate / selected
        for key, value in [('current', selected), ('candidate', candidate), ('ratio', ratio)]:
            assert value == Fraction(original[key]['numerator'], original[key]['denominator'])
        assert original['passed'] == (ratio <= Fraction(105, 100))
        rows.append(original)
    return dict(closure=INPUTS[(base / 'closed.json').relative_to(ROOT).as_posix()],
        admitted=False, products=analysis['products'], rows=rows, controls=analysis['controls'],
        gates=analysis['gates'], processes=processes)


def diagnostic():
    base, proof = closed('parakeet-pad-runtime-diagnostic-amd-20260923',
        '2e718d3095eaea8ec79fb263d5ca504564a5e5a83e986ef5535c53505b3864de')
    assert proof['diagnostic_only']
    original = read(base / 'analysis.json', proof['files']['analysis.json'])
    # Reuse only pure marker reconciliation, never the closed campaign's main().
    module = ROOT / 'tests/parakeet/pad-runtime-diagnostic-amd'
    for name in ('audit.py', 'protocol.py', 'prepare.py', 'census.py'):
        INPUTS[(module / name).relative_to(ROOT).as_posix()] = pin(module / name)
    sys.path.insert(0, str(module))
    from audit import reconcile
    assoc_module = ROOT / 'tests/parakeet/pad-runtime-diagnostic-results'
    INPUTS[(assoc_module / 'associations.py').relative_to(ROOT).as_posix()] = pin(assoc_module / 'associations.py')
    sys.path.insert(0, str(assoc_module))
    from associations import associations
    roles = {}
    for role in ('current', 'candidate'):
        def bound(name):
            return read(base / name, proof['files'][name])
        value = bound(f'collected/{role}-capture/result.json')
        summary = bound(f'collected/{role}-export/events/summary.json')
        name = f'collected/{role}-export/events/events.jsonl'
        path = base / name
        assert pin(path) == proof['files'][name]
        INPUTS[path.relative_to(ROOT).as_posix()] = proof['files'][name]
        with path.open(encoding='utf8') as stream:
            events = [json.loads(line) for line in stream]
        report = reconcile(value, events, summary)
        assert report == original['reports'][role]
        assoc = associations(report['calls'], events)
        loads = [load for load in assoc['loads']
                 if load['namespace'] == 'Lokad.Onnx.CPUExecutionProvider'
                 and load['method'] in ('Pad', 'PadCore', 'PadDispatch')]
        cases = []
        for row in value['rows']:
            index = row['index']
            calls = report['calls'][780 * index:780 * (index + 1)]
            measured = calls[600:]
            start = measured[0]['begin_ms']
            end = measured[-1]['end_ms']
            tiers = {}
            for method in ('Pad', 'PadCore', 'PadDispatch'):
                method_loads = [load for load in loads if load['method'] == method]
                tiers[method] = dict(
                    full_optimized_before_measurement=any(load['tier'] == 'OptimizedTier1'
                        and load['ms'] < start for load in method_loads),
                    available_before_measurement=[load for load in method_loads if load['ms'] < start],
                    loads_during_measurement=[load for load in method_loads if start <= load['ms'] <= end])
            # Paired suspension events can overlap markers. Keep this separate from
            # the public-operation clocks and never subtract it from the result.
            suspension = sum(max(0, min(call['end_ms'], p['end_ms']) -
                max(call['begin_ms'], p['start_ms']))
                for call in measured for p in assoc['suspensions'])
            cases.append(dict(index=index, name=row['name'], shape=row['shape'], pads=row['pads'],
                first_ms=calls[0]['begin_ms'], measured_start_ms=start, end_ms=end,
                mean_ms=statistics.mean(c['wall_ms'] for c in measured),
                measured_blocks_ms=[statistics.mean(c['wall_ms'] for c in measured[i:i + 60])
                                    for i in (0, 60, 120)],
                marker_suspension_overlap_ms=suspension,
                median_counter_allocation_bytes=statistics.median(c['allocated_bytes'] for c in measured),
                compilation_availability=tiers))
        roles[role] = dict(events=len(events), marker_count=report['markers'],
            first_ms=report['calls'][0]['begin_ms'], last_ms=report['calls'][-1]['end_ms'],
            cases=cases, loads=loads,
            tiered_compilation_event_count=sum('TieredCompilation' in e['name'] for e in events))
    return roles


def main():
    destination = OUT / 'observations-20260926.json'
    assert not destination.exists(), 'Do not overwrite a completed review.'
    traced = diagnostic()
    screens = dict(
        dispatch=screen('parakeet-pad-dispatch-screen-amd-20260923',
            '8e757188ca7a29c9c78a9fdc1803eb64d16dab18c476e0e433419eafd5245d73'),
        first_use=screen('parakeet-pad-first-use-screen-amd-20260923',
            '7ec1fbae97f5ddae7d93af3eab9e002b78c2acc9aa349147b8ddc33fb174ed9f'))
    composition = ROOT / 'tests/parakeet/pad-first-use-results/composition-20260923.json'
    comp = read(composition)
    assert comp['passed'] and comp['only_padcore_implementation_flag_changed']
    assert comp['padcore_flag_before'] == 0 and comp['padcore_flag_after'] == 512
    assert comp['m47_helper_body_exact'] and comp['original_padcore_body_exact']
    for name in ('tests/parakeet/pad-dispatch-source/Zzz.LastAxisPadDispatch.cs',
                 'src/Lokad.Onnx/CPUExecutionProvider.Shape.cs',
                 'artifacts/parakeet-memory-source-review-20260923/ort/onnxruntime/core/providers/cpu/tensor/pad.cc'):
        INPUTS[name] = pin(ROOT / name)
    result = dict(passed=True, diagnostic_only=True, admitted=False, new_inference=False,
        original_screens_remain_rejected=True, inputs=INPUTS,
        generator=pin(Path(__file__)), trace=traced, screens=screens,
        limitations=['Method loads establish availability, not every executed instruction.',
            'No compilation trace exists for the first-use screen in the reviewed evidence.',
            'No tiering pause/resume events were captured in the old runtime diagnostic.',
            'Instrumented allocation counters do not isolate product allocations.',
            'The ORT padding algorithm is source-derived; no native Pad leaf is independently sampled.'])
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(json.dumps(dict(passed=True, output=pin(destination), inputs=len(INPUTS),
        complete_diagnostic_calls=18720, original_screen_calls=74880,
        both_original_screens_rejected=True, new_inference=False)))


if __name__ == '__main__':
    main()
