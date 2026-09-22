"""Independently close both actual product instruction widths and every case."""
import ast
import importlib.util
import json
from run import BASE, ROOT, prepared
from protocol import LIMITS, check_sample, pin, read, save


def raw_check(result, lanes):
    path = ROOT/'tests/pyannote/blocked-spatial-raw-graphs/audit.py'
    tree = ast.parse(path.read_text()); function, = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'check']
    namespace = dict(CORE=read(BASE/'payload/payload.json')['core']['sha256'])
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['check'](result, lanes)


def local_resources():
    path = ROOT/'tests/parakeet/portable-models/common.py'
    spec = importlib.util.spec_from_file_location('product_local_resource_auditor', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    value = read(BASE/'local-inputs.json'); module.verify(value['files'])
    return module.resources(BASE, 'local-controller.json', dict(restore=(8, 8, 900, False), build=(8, 8, 900, False), **{'layers-local': (12, 8, 900, True)}))


def layers_check(result, lanes):
    original = read(BASE/'payload/windows-layers.json'); fixture = read(BASE/'payload/fixtures/result.json')
    assert result['passed'] and result['cases'] == len(result['observations']) == 108
    assert result['eligible'] == 96 and result['fallback'] == 12
    assert result['failed'] == result['differences'] == result['native_failures'] == 0
    assert result['values'] == 119823360 and result['observations'] == original['observations']
    assert result['maximum'] == original['maximum'] <= 1e-4
    assert result['read_only_operands'] and result['layer_graphs'] == 108 and result['graph_retained_bytes'] == 3*21086208
    assert result['prepared_weights'] == 32 and result['prepared_bytes'] == original['prepared_bytes']
    assert result['lanes'] == lanes and len(result['graph_dispatch']) == 216
    for i, row in enumerate(result['graph_dispatch']):
        call = fixture['calls'][i//2]
        assert (row['name'], row['index'], row['form'], row['eligible'], row['repeat']) == (call['case'], call['index'], call['form'], call['eligible'], i % 2)
        x, y = call['input']['shape'], call['output']['shape']
        if row['eligible']:
            assert row['scratch_bytes'] == (x[1]*(x[2]+2)*(x[3]+2)+y[1]*y[2]*y[3])*4
            assert row['prepared_bytes'] == call['weights']['bytes']
        else: assert row['prepared_bytes'] == 0 and row['scratch_bytes'] > 0
        residual = call['residual'] is not None
        expected = ['ConvRelu' if call['relu'] and not residual else 'Conv']
        if residual: expected.append('AddRelu' if call['relu'] else 'Add')
        assert row['nodes'] == expected
    return dict(passed=True, cases=108, values=119823360, graph_calls=216, native_maximum=result['maximum'], lanes=lanes)


def main():
    spec = prepared(); assert not (BASE/'closed.json').exists(); local = local_resources()
    payload = read(BASE/'payload/payload.json'); collected = BASE/'collected'; receipt = read(collected/'collection.json')
    transfer = read(BASE/'collection-transfer.json')
    assert transfer['passed'] and transfer['archive'] == pin(BASE/'results.tar.gz') and transfer['receipt'] == pin(collected/'collection.json')
    assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
    for name, wanted in receipt['files'].items(): assert pin(collected/name) == wanted, name
    for name, wanted in payload['files'].items(): assert pin(BASE/'payload'/name) == wanted, name
    state = read(collected/'identity.json'); assert state['complete'] and state['code'] == 0
    assert state['supervisor'] == read(BASE/'deployment.json') and state['boot_time'] == 1789634288.0
    assert [r['name'] for r in state['runs']] == payload['jobs'] == ['raw-256', 'raw-512', 'layers-256', 'layers-512']
    assert receipt['identities'] == [state['supervisor']]+[dict(pid=int(p), birth=b) for r in state['runs'] for p,b in r['members'].items()]
    resources = []; reports = {}
    for row in state['runs']:
        mode, width = row['name'].split('-'); lanes = int(width)//32
        assert row['complete'] and row['code'] == 0 and row['seconds'] < LIMITS['seconds']
        assert row['preflight']['available'] >= LIMITS['preflight_available'] and row['preflight']['tmpfs'] >= LIMITS['preflight_tmpfs']
        assert row['preflight'] == row['preflight_observations'][-1] == read(collected/(row['name']+'-preflight.json'))[-1]
        samples = [json.loads(s) for s in (collected/'logs'/(row['name']+'.jsonl')).read_text().splitlines()]
        assert len(samples) == row['samples'] > 0 and max(s['rss'] for s in samples) == row['peak_rss']
        for sample in samples:
            check_sample(sample)
            for member in sample['members']: assert row['members'][str(member['pid'])] == member['birth']
        result = read(collected/row['name']/'result.json')
        assert result['passed'] and result['pid'] == row['child']['pid'] and result['runtime'] == '10.0.8'
        assert result['core'] == payload['core']['sha256'] and result['executable'] == payload['consumers'][mode]['sha256']
        assert result['flags'] == (['DOTNET_EnableAVX512F'] if width == '256' else []) and result['avx512'] == (width == '512')
        reports[row['name']] = raw_check(result, lanes) if mode == 'raw' else layers_check(result, lanes)
        original = read(BASE/'payload'/('windows-raw.json' if mode == 'raw' else 'windows-layers.json'))
        assert result['observations'] == original['observations']
        resources.append(dict(name=row['name'], samples=len(samples), peak_rss=row['peak_rss'], seconds=row['seconds']))
    assert state['ended']-state['started'] < 4*3600
    analysis = dict(passed=True, reports=reports, resources=resources, local_resources=local['resources'],
        samples=sum(r['samples'] for r in resources), peak_rss=max(r['peak_rss'] for r in resources),
        core=payload['core'], consumers=payload['consumers'], no_performance_measurement=True)
    save(BASE/'analysis.json', analysis)
    files = {p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file() and not {'obj', 'packages'}.intersection(p.relative_to(BASE).parts)}
    save(BASE/'closed.json', dict(passed=True, files=files, local_inputs=spec['files'], local_identities=local['identities'],
        remote_terminal=receipt['identities'], analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'), **analysis)))


if __name__ == '__main__': main()
