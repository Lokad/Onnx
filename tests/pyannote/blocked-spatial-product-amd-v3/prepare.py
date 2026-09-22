"""Reuse built consumers; explicitly review the compiler's changed string cache."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
import traceback

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/pyannote-blocked-spatial-product-amd-v3-20260922'
V2 = ROOT/'artifacts/pyannote-blocked-spatial-product-amd-v2-20260922'
FAILED = ROOT/'artifacts/pyannote-blocked-spatial-product-amd-20260922'
PRODUCT = ROOT/'artifacts/pyannote-blocked-spatial-composition-v3-20260922'
RAW = ROOT/'artifacts/pyannote-blocked-spatial-raw-graphs-20260922'
MONITOR = ROOT/'tests/parakeet/packing-budgets/common.py'
CORE = '3c2f16b08856426d3dfeff07f1638dd76cee7f06b65bbee230e8e0789679206f'
spec = importlib.util.spec_from_file_location('v3_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec); spec.loader.exec_module(monitor); monitor.BASE = BASE
from protocol import pin, read, save, LIMITS, check_sample


def review():
    observations = []
    old_field = '<PrivateImplementationDetails>::System.String[] D1FA81A5648BDAFEA4032DF441DEE909E3129C285714128E1F7B9752A4869A3D_B11'
    new_field = '<PrivateImplementationDetails>::System.String[] 306C0EFD441B2EC3EF12970F43FA292316CDF1D634B74A7327C6B89DCCC12D1D_B11'
    for mode, assembly, owner in [('raw', 'Lokad.Onnx.Backend.Tests.dll', 'Probe'), ('layers', 'LayerGraphs.dll', 'ModelProbe')]:
        value = read(V2/(mode+'-instructions.json')); assert value['inventory_complete']
        row, = value['observations']; assert row['assembly'] == assembly and row['public_surface_equal']
        assert not row['removed'] and not row['added'] and row['unchanged_methods'] == row['methods']-1
        method, = row['differences']; assert method.startswith(owner+'::Main::')
        before, after = json.loads(row['normalized_methods'][method]), json.loads(row['candidate_methods'][method])
        changed = []
        for left, right in zip(before['instructions'], after['instructions'], strict=True):
            if left == right: continue
            assert left['offset'] == right['offset'] and left['opcode'] == right['opcode']
            if left['opcode'] == 'ldstr':
                assert (left['operand'], right['operand']) == ('DOTNET_EnableAVX512F', 'DOTNET_EnableAVX512')
            else:
                assert left['opcode'] in ['ldsfld', 'stsfld']
                assert (left['operand'], right['operand']) == (old_field, new_field)
            changed.append(dict(before=dict(left), after=dict(right)))
            left['operand'] = right['operand']
        assert before == after and sorted(c['before']['opcode'] for c in changed) == ['ldsfld', 'ldstr', 'ldstr', 'stsfld']
        assert row['before_sha256'] == pin(FAILED/'payload/runtime'/assembly)['sha256']
        assert row['after_sha256'] == pin(V2/'source'/mode/'bin/Release/net10.0'/assembly)['sha256']
        observations.append(dict(mode=mode, methods=row['methods'], unchanged=row['unchanged_methods'], changed=method, reviewed_instructions=changed))
    return dict(passed=True, observations=observations, product_unchanged=True, core=pin(PRODUCT/'runtime/Lokad.Onnx.dll'))


def previous_closed():
    for folder, name in [(FAILED, 'failure-closed.json'), (V2, 'failure-closed.json')]:
        proof = read(folder/name); assert proof['retained_failure'] and not proof['passed']
        for name, wanted in proof['files'].items(): assert pin(folder/name) == wanted, name
        for identity in proof['local_identities']: monitor.terminal(identity)
    return read(FAILED/'deployment.json')


def close_review_failure():
    if (V2/'failure-closed.json').exists(): return
    state = read(V2/'local-controller.json')
    assert state['complete'] and state['code'] == 1 and 'review' in state['error'] and 'AssertionError' in state['error']
    jobs = [m+'-'+j for m in ['raw', 'layers', 'bridge'] for j in ['restore', 'build']]+['raw-inventory', 'layers-inventory']
    assert [r['name'] for r in state['runs']] == jobs and not (V2/'prepared.json').exists()
    monitor.verify(read(V2/'local-inputs.json')['files'])
    identities = [state['supervisor']]; resources = []
    for row in state['runs']:
        assert row['complete'] and row['code'] == 0 and row['preflight']['available'] >= 8*1024**3
        identities += [dict(pid=int(p), birth=b) for p,b in row['members'].items()]
        samples = [json.loads(s) for s in (V2/'logs'/(row['name']+'.samples.jsonl')).read_text().splitlines()]
        assert len(samples) == row['samples'] > 0 and max(s['rss'] for s in samples) == row['peak_rss']
        assert samples[-1]['seconds'] <= row['seconds'] < 900
        for s in samples:
            assert s['rss'] < 8*1024**3 and s['available'] >= 1024**3 and s['disk'] >= 20*1024**3 and s['output_bytes'] <= 1024**3
            assert s['rss'] == sum(p['rss'] for p in s['members'])
            assert all(p['affinity'] == [2] and row['members'][str(p['pid'])] == p['birth'] for p in s['members'])
        resources.append(dict(name=row['name'], samples=len(samples), peak_rss=row['peak_rss']))
    for identity in identities: monitor.terminal(identity)
    analysis = dict(passed=False, retained_failure=True, numerical_cases_run=False, target_workers_run=False,
        reason='Verifier omitted the compiler-generated cached string-array field rename; builds and inventories passed.',
        review=review(), resources=resources)
    save(V2/'failure-analysis.json', analysis)
    files = {p.relative_to(V2).as_posix(): pin(p) for p in V2.rglob('*') if p.is_file() and not {'obj', 'packages'}.intersection(p.relative_to(V2).parts)}
    save(V2/'failure-closed.json', dict(passed=False, retained_failure=True, files=files, local_identities=identities, analysis=pin(V2/'failure-analysis.json')))


def prepare():
    assert not BASE.exists(); close_review_failure(); owner = previous_closed()
    BASE.mkdir(); (BASE/'logs').mkdir(); (BASE/'output').mkdir()
    inputs = dict(read(V2/'local-inputs.json')['files'])
    for p in [MONITOR, V2/'failure-closed.json', FAILED/'failure-closed.json', ROOT/'tests/pyannote/blocked-spatial-raw-graphs/audit.py', *TOOLS.iterdir()]:
        if p.is_file(): inputs[p.as_posix()] = pin(p)
    inputs.update({p.as_posix(): pin(p) for mode in ['raw', 'layers'] for p in (V2/'source'/mode/'bin/Release/net10.0').iterdir() if p.is_file()})
    save(BASE/'local-inputs.json', dict(files=inputs)); save(BASE/'instruction-review.json', review())
    own = monitor.psutil.Process(); state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    try:
        for mode, executable in [('raw', 'Lokad.Onnx.Backend.Tests.dll'), ('layers', 'LayerGraphs.dll')]:
            args = ['dotnet', V2/'source'/mode/'bin/Release/net10.0'/executable, '256']
            if mode == 'layers': args.append(FAILED/'payload/fixtures')
            args.append(BASE/'output'/(mode+'.json'))
            monitor.worker(state, BASE/'local-controller.json', mode+'-local', args, ROOT, [0], 12, 8, 900, False, BASE/'output')
            result = read(BASE/'output'/(mode+'.json'))
            expected = read(RAW/'output/256.json' if mode == 'raw' else FAILED/'output/layers.json')
            assert result['passed'] and result['core'] == CORE and not result['flags']
            for key in (['observations', 'graph_cases', 'supplemental'] if mode == 'raw' else ['observations', 'graph_dispatch']): assert result[key] == expected[key], (mode, key)
            print(mode, 'passed', flush=True)
        monitor.verify(inputs); previous_closed()
        payload = BASE/'payload'; payload.mkdir(); (payload/'tools').mkdir(); (payload/'fixtures').mkdir()
        shutil.copytree(PRODUCT/'runtime', payload/'runtime')
        assert pin(payload/'runtime/Lokad.Onnx.dll')['sha256'] == CORE
        shutil.copy2(FAILED/'payload/fixtures/result.json', payload/'fixtures/result.json')
        for name in ['remote.py', 'protocol.py']: shutil.copy2(TOOLS/name, payload/'tools'/name)
        for mode, name in [('raw', 'Lokad.Onnx.Backend.Tests'), ('layers', 'LayerGraphs')]:
            for suffix in ['dll', 'deps.json', 'runtimeconfig.json']: shutil.copy2(V2/'source'/mode/'bin/Release/net10.0'/(name+'.'+suffix), payload/'runtime'/(name+'.'+suffix))
            shutil.copy2(BASE/'output'/(mode+'.json'), payload/('windows-'+mode+'.json'))
        shutil.copy2(ROOT/'.agent/m17-pyannote-blocked-product-20260922.md', payload/'prospective-plan.md')
        old = read(FAILED/'payload/payload.json'); external = dict(old['external'])
        remote_fixtures = '/dev/shm/lokad-pyannote-blocked-spatial-product-20260922/fixtures'
        for p in (FAILED/'payload/fixtures').iterdir():
            if p.is_file(): external[remote_fixtures+'/'+p.name] = pin(p)
        manifest = dict(passed=True, limits=LIMITS, previous_owner=owner, boot_time=1789634288.0, interpreter=old['interpreter'],
            external=external, fixture_directory=remote_fixtures, core=pin(payload/'runtime/Lokad.Onnx.dll'),
            consumers=dict(raw=pin(payload/'runtime/Lokad.Onnx.Backend.Tests.dll'), layers=pin(payload/'runtime/LayerGraphs.dll')),
            jobs=['raw-256', 'raw-512', 'layers-256', 'layers-512'], files={p.relative_to(payload).as_posix(): pin(p) for p in payload.rglob('*') if p.is_file()},
            scope='Correct switch and exact generated-field review; unchanged actual product; both AMD widths; no timing')
        save(payload/'payload.json', manifest)
        for p in TOOLS.glob('*.py'): ast.parse(p.read_text(), str(p))
        with tarfile.open(BASE/'payload.tar.gz', 'w:gz') as archive:
            for p in sorted(payload.rglob('*')):
                if p.is_file(): archive.add(p, arcname=p.relative_to(payload).as_posix(), recursive=False)
        save(BASE/'prepared.json', dict(passed=True, files={p.relative_to(ROOT).as_posix(): pin(p) for p in [BASE/'local-inputs.json', BASE/'instruction-review.json', V2/'failure-closed.json', *TOOLS.iterdir()] if p.is_file()}, payload=pin(payload/'payload.json'), archive=pin(BASE/'payload.tar.gz')))
        state['code'] = 0; print(json.dumps(dict(payload=pin(payload/'payload.json'), archive=pin(BASE/'payload.tar.gz'))), flush=True)
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); raise
    finally:
        state['complete'] = True; save(BASE/'local-controller.json', state)


if __name__ == '__main__': prepare()
