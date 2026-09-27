"""Close complete original-decoder evidence without admitting a performance change."""
import json
from pathlib import Path
from protocol import JOBS, LIMITS, PROVIDERS, pin, read, save, check_sample
from prepare import ROOT, BASE, QUALIFIED, previous_closed
from run import REMOTE, prepared
from checks import result
from events import reconcile, stacks, method_records
import remote


def main():
    assert not (BASE/'closed.json').exists() and not (BASE/'analysis.json').exists()
    previous_closed(); prepared()
    folder = BASE/'collected'
    receipt, state, payload = read(folder/'collection.json'), read(folder/'identity.json'), read(BASE/'payload.json')
    assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
    assert receipt['payload'] == pin(BASE/'payload.json') == pin(folder/'payload.json')
    transfer = read(BASE/'collection-transfer.json')
    assert transfer['passed'] and transfer['archive'] == pin(BASE/'results.tar.gz') and transfer['receipt'] == pin(folder/'collection.json')
    for name, wanted in receipt['files'].items(): assert pin(folder/name) == wanted, name
    assert state['complete'] and state['code'] == 0 and state['boot_time'] == 1789634288.0
    assert state['supervisor'] == read(BASE/'deployment.json')
    assert state['ended'] - state['started'] < 4*3600
    assert [r['name'] for r in state['runs']] == payload['jobs'] == JOBS
    assert payload['limits'] == LIMITS
    assert receipt['identities'] == [state['supervisor']] + [dict(pid=int(p), birth=b) for r in state['runs'] for p, b in r['members'].items()]
    assert payload['root_qualification'] == pin(QUALIFIED/'closed.json')
    assert payload['product'] == read(QUALIFIED/'analysis.json')['built']
    resources, runs = [], {r['name']: r for r in state['runs']}
    remote.BASE = Path(REMOTE)
    for row in state['runs']:
        assert row['complete'] and row['code'] == 0 and row['seconds'] < LIMITS['seconds']
        assert all(code == 0 for code in row['exitcodes'].values())
        assert row['preflight']['available'] >= LIMITS['preflight_available'] and row['preflight']['tmpfs'] >= LIMITS['preflight_tmpfs']
        command, _, cpu = remote.command_for(row['name'], payload)
        assert row['commands']['worker'] == list(map(str, command))
        assert row['processes']['worker']['affinity'] == [cpu]
        expected = {'worker', 'collector'} if row['name'] == 'trace-capture' else {'worker'}
        assert set(row['processes']) == set(row['exitcodes']) == expected
        if row['name'] == 'trace-capture':
            assert row['processes']['collector']['affinity'] == [0]
            expected_command = [remote.DOTNET, Path(REMOTE)/'tracer/dotnet-trace.dll', 'collect', '--process-id',
                row['processes']['worker']['pid'], '--providers', PROVIDERS, '--buffersize', '64',
                '--duration', '00:00:15:00', '--output', Path(REMOTE)/'trace-capture/capture.nettrace']
            assert row['commands']['collector'] == list(map(str, expected_command))
        for owner in row['processes'].values():
            assert row['members'][str(owner['pid'])] == owner['birth']
            assert row['affinities'][str(owner['pid'])] == owner['affinity']
        samples = [json.loads(line) for line in (folder/'logs'/(row['name']+'.jsonl')).read_text().splitlines()]
        assert len(samples) == row['samples'] and samples
        for sample in samples:
            check_sample(sample)
            for member in sample['members']:
                assert row['members'][str(member['pid'])] == member['birth']
                assert row['affinities'][str(member['pid'])] == member['expected_affinity']
        assert max(s['rss'] for s in samples) == row['peak_rss']
        resources.append(dict(name=row['name'], samples=len(samples), peak_rss=row['peak_rss'], seconds=row['seconds']))
    built = read(folder/'built.json'); assert built['passed']
    for name, wanted in built['files'].items(): assert pin(folder/name) == wanted, name
    assert pin(folder/'runtime/Lokad.Onnx.dll') == payload['product']['Lokad.Onnx.dll']
    assert pin(folder/'runtime/ParakeetDecoderObservation.dll') == built['consumer']
    reports = {}
    for name in ['control-run', 'trace-capture']:
        reports[name] = result(folder, name, payload, runs[name]['processes']['worker'], built)
        assert reports[name] == read(folder/name/'review.json')
    trace = read(folder/'trace-capture/result.json')
    summary = read(folder/'trace-export/events/summary.json')
    assert summary['input_sha256'] == pin(folder/'trace-capture/capture.nettrace')['sha256']
    assert summary['exporter_pid'] == runs['trace-export']['processes']['worker']['pid']
    assert pin(folder/'export-runtime/DispatchEventsExport.dll')['sha256'] == '496437aefd28e722a3d759c7de1cfb7288c0eff310225a6fbb0b80274d16903d'
    events = [json.loads(line) for line in (folder/'trace-export/events/events.jsonl').read_text().splitlines()]
    joined = reconcile(trace, events, summary)
    path, = (folder/'trace-stacks').glob('*.speedscope.json')
    stack_result = stacks(read(path), joined['intervals'], trace['native_thread'])
    value = dict(passed=True, diagnostic_only=True, performance_admitted=False, product_changed=False,
        root_qualification=payload['root_qualification'], product=payload['product'], consumer=built['consumer'],
        reports=reports, events=joined, stack_intervals=stack_result, method_events=method_records(events),
        mapping=trace['mapping'], operands=trace['operands'], resources=resources,
        per_call_jit_version_proved=False, per_node_allocation_or_copy_proved=False,
        scope='One original decoder fixture; decide the prepared-weight route from actual mapping and named method samples.')
    save(BASE/'analysis.json', value)
    files = {p.relative_to(BASE).as_posix(): pin(p) for p in folder.rglob('*') if p.is_file()}
    for name in ['analysis.json', 'prepared.json', 'staged.json', 'extracted.json', 'stage-started.json', 'payload.json',
                 'deployment.json', 'collection-transfer.json', 'results.tar.gz', 'payload.tar.gz']:
        files[name] = pin(BASE/name)
    save(BASE/'closed.json', dict(passed=True, diagnostic_only=True, performance_admitted=False,
        analysis=pin(BASE/'analysis.json'), files=files, remote_terminal=True))
    print(json.dumps(dict(passed=True, closed=pin(BASE/'closed.json'), prepared_mapping_present=trace['mapping']['present'],
                         calls=2560, control_calls=6, resource_observations=sum(r['samples'] for r in resources),
                         performance_admitted=False)))


if __name__ == '__main__': main()
