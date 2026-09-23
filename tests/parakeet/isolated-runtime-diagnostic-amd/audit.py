"""Reconcile every matrix call and runtime event without a speed score."""
import base64
import collections
import json
import math
import struct
from protocol import JOBS, LIMITS, PROVIDERS, DISASM, check_sample, pin, read, save
from prepare import ROOT, BASE, SCREEN, previous_closed
from checks import reconcile, codegen, codegen_bodies


def main():
    previous_closed(); assert not (BASE / 'closed.json').exists()
    from run import prepared
    prepared(); folder = BASE / 'collected'
    receipt = read(folder / 'collection.json'); state = read(folder / 'identity.json'); payload = read(BASE / 'payload.json')
    assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None and receipt['payload'] == pin(BASE / 'payload.json')
    for n, v in receipt['files'].items(): assert pin(folder / n) == v, n
    assert state['complete'] and state['code'] == 0 and state['supervisor'] == read(BASE / 'deployment.json') and state['boot_time'] == 1789634288.0
    assert [r['name'] for r in state['runs']] == payload['jobs'] == JOBS and state['ended'] - state['started'] < 4 * 3600
    assert receipt['identities'] == [state['supervisor']] + [dict(pid=int(p), birth=b) for r in state['runs'] for p, b in r['members'].items()]
    transfer = read(BASE / 'collection-transfer.json')
    assert transfer['passed'] and transfer['archive'] == pin(BASE / 'results.tar.gz') and transfer['receipt'] == pin(folder / 'collection.json')
    capture = read(folder / 'evidence/capture.json')
    assert capture == read(SCREEN / 'bundle/fixtures/result.json')
    for n, v in payload['files'].items():
        if (folder / n).is_file(): assert pin(folder / n) == v, n
    resources = 0; peak = 0; runs = {r['name']: r for r in state['runs']}
    for row in state['runs']:
        assert row['complete'] and row['code'] == 0 and row['seconds'] < LIMITS['seconds'] and all(c == 0 for c in row['exitcodes'].values())
        assert row['preflight']['available'] >= LIMITS['preflight_available'] and row['preflight']['tmpfs'] >= LIMITS['preflight_tmpfs']
        cpu = 0 if row['name'].endswith('-export') or row['name'] == 'tracer-version' else 2
        assert row['processes']['worker']['affinity'] == [cpu]
        if row['name'].endswith('-capture'):
            assert set(row['processes']) == {'worker', 'collector'} and row['processes']['collector']['affinity'] == [0]
            cmd = row['commands']['collector']; assert cmd[cmd.index('--providers') + 1] == PROVIDERS
            assert int(cmd[cmd.index('--process-id') + 1]) == row['processes']['worker']['pid']
        else: assert set(row['processes']) == {'worker'}
        if row['name'].endswith('-codegen'):
            assert row['commands']['worker'][:2] == ['/usr/bin/env', 'DOTNET_JitDisasm=' + DISASM]
        samples = [json.loads(line) for line in (folder / 'logs' / (row['name'] + '.jsonl')).read_text().splitlines()]
        assert samples and len(samples) == row['samples'] and max(s['rss'] for s in samples) == row['peak_rss']
        for sample in samples:
            check_sample(sample)
            for m in sample['members']:
                assert row['members'][str(m['pid'])] == m['birth'] and row['affinities'][str(m['pid'])] == m['expected_affinity']
        gaps = [samples[0]['seconds']] + [b['seconds'] - a['seconds'] for a, b in zip(samples, samples[1:])] + [row['seconds'] - samples[-1]['seconds']]
        assert all(0 <= gap < 10 for gap in gaps)
        resources += len(samples); peak = max(peak, row['peak_rss'])
    built = read(folder / 'built.json'); assert built['passed']
    assert (folder / 'logs/sdk-version.stdout').read_text().strip().endswith('10.0.204')
    for n, v in built['files'].items(): assert pin(folder / n) == v, n
    assert built['exporter'] == payload['exporter'] == pin(folder / 'export-runtime/DispatchEventsExport.dll')
    assert built['capture_consumer'] == payload['capture_consumer']
    reports = {}
    for sequence, role in enumerate(['current', 'candidate']):
        assert pin(folder / 'runtimes' / role / 'ParakeetDispatchEvents.dll') == built['capture_consumer']
        assert pin(folder / 'runtimes' / role / 'ParakeetIsolatedCodegen.dll') == built['consumer']
        for n, v in payload['products'][role].items(): assert pin(folder / 'runtimes' / role / n) == v, n
        value = read(folder / (role + '-capture/result.json'))
        ready = read(folder / (role + '-capture/ready.json')); enabled = read(folder / (role + '-capture/collector-enabled.json'))
        assert value['pid'] == runs[role + '-capture']['processes']['worker']['pid'] == ready['pid'] == enabled['pid']
        assert value['nativeThread'] == ready['native_thread'] and ready['counter'] < enabled['counter'] < value['rows'][0]['clocks'][0]['marker']
        assert value['runtime'] == '10.0.8' and value['flags'] == {} and value['role'] == role and value['sequence'] == sequence
        assert value['core_sha256'] == payload['products'][role]['Lokad.Onnx.dll']['sha256'] and value['assembly'] == payload['capture_consumer']['sha256']
        original = read(folder / 'evidence' / ('original-' + role + '.json'))
        for a, b in zip(value['rows'], original['rows'], strict=True):
            for key in ['index', 'name', 'node', 'm', 'reduction', 'columns', 'output', 'exact', 'inputs', 'guards']:
                assert a[key] == b[key], key
        summary = read(folder / (role + '-export/events/summary.json'))
        assert summary['input_sha256'] == pin(folder / (role + '-capture/capture.nettrace'))['sha256']
        assert summary['runtime'] == '10.0.8' and summary['exporter_pid'] == runs[role + '-export']['processes']['worker']['pid']
        events = [json.loads(line) for line in (folder / (role + '-export/events/events.jsonl')).read_text().splitlines()]
        reports[role] = reconcile(value, events, summary, capture)
        generated = read(folder / (role + '-codegen/result.json'))
        assert generated['pid'] == runs[role + '-codegen']['processes']['worker']['pid']
        assert generated['runtime'] == '10.0.8' and generated['role'] == role and generated['sequence'] == sequence
        assert generated['assembly'] == built['consumer']['sha256'] and generated['core_sha256'] == payload['products'][role]['Lokad.Onnx.dll']['sha256']
        codegen(generated, capture)
        bodies = codegen_bodies((folder / 'logs' / (role + '-codegen.stdout')).read_text(), role)
        assert bodies == read(folder / (role + '-codegen/bodies.json'))
        reports[role]['codegen_bodies'] = bodies
        reports[role]['caller_probes'] = generated['callerProbes']
    save(BASE / 'analysis.json', dict(passed=True, diagnostic_only=True, root_product_changed=False, resources=resources,
        peak_rss=peak, products=payload['products'], codegen_consumer=built['consumer'], capture_consumer=payload['capture_consumer'], exporter=built['exporter'], reports=reports))
    files = {p.relative_to(BASE).as_posix(): pin(p) for p in folder.rglob('*') if p.is_file()}
    for n in ['analysis.json', 'prepared.json', 'staged.json', 'payload.json', 'deployment.json', 'collection-transfer.json', 'results.tar.gz', 'payload.tar.gz']:
        files[n] = pin(BASE / n)
    save(BASE / 'closed.json', dict(passed=True, diagnostic_only=True, root_product_changed=False, files=files))
    print(json.dumps(dict(closed=pin(BASE / 'closed.json'), resources=resources, peak_rss=peak,
        reports={k: {n: len(v) if isinstance(v, list) else v for n, v in r.items()} for k, r in reports.items()})))


if __name__ == '__main__': main()
