"""Correct only the runtime-switch consumer literals; preserve product and gates."""
import ast
import difflib
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
import traceback
import urllib.request
from protocol import LIMITS, pin, read, save
from close_failure import FAILED

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/pyannote-blocked-spatial-product-amd-v2-20260922'
PRODUCT = ROOT/'artifacts/pyannote-blocked-spatial-composition-v3-20260922'
RAW = ROOT/'artifacts/pyannote-blocked-spatial-raw-graphs-20260922'
BRIDGE = ROOT/'artifacts/pyannote-combined-consumers-20260922/bridge'
CORE = '3c2f16b08856426d3dfeff07f1638dd76cee7f06b65bbee230e8e0789679206f'
FEED = ROOT/'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
MONITOR = ROOT/'tests/parakeet/packing-budgets/common.py'
REMOTE_FIXTURES = '/dev/shm/lokad-pyannote-blocked-spatial-product-20260922/fixtures'
spec = importlib.util.spec_from_file_location('corrected_product_local_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec); spec.loader.exec_module(monitor); monitor.BASE = BASE


def previous_closed():
    proof = read(FAILED/'failure-closed.json'); assert proof['retained_failure'] and not proof['passed']
    for name, wanted in proof['files'].items(): assert pin(FAILED/name) == wanted, name
    for identity in proof['local_identities']: monitor.terminal(identity)
    receipt = read(FAILED/'collected/collection.json'); assert receipt['terminal'] and receipt['input_error'] is None
    assert not read(FAILED/'failure-analysis.json')['numerical_cases_run']
    return read(FAILED/'deployment.json')


def review():
    observations = []
    for mode, assembly, owner in [('raw', 'Lokad.Onnx.Backend.Tests.dll', 'Probe'), ('layers', 'LayerGraphs.dll', 'ModelProbe')]:
        value = read(BASE/(mode+'-instructions.json')); assert value['inventory_complete']
        row, = value['observations']; assert row['assembly'] == assembly and row['public_surface_equal']
        assert not row['removed'] and not row['added'] and row['unchanged_methods'] == row['methods']-1
        method, = row['differences']; assert method.startswith(owner+'::Main::')
        before, after = row['normalized_methods'][method], row['candidate_methods'][method]
        assert before.count('DOTNET_EnableAVX512F') == 2 and before.replace('DOTNET_EnableAVX512F', 'DOTNET_EnableAVX512') == after
        assert row['before_sha256'] == pin(FAILED/'payload/runtime'/assembly)['sha256']
        assert row['after_sha256'] == pin(BASE/'source'/mode/'bin/Release/net10.0'/assembly)['sha256']
        observations.append(dict(mode=mode, methods=row['methods'], unchanged=row['unchanged_methods'], changed=method, only_two_flag_literals_changed=True))
    return dict(passed=True, observations=observations, product_unchanged=True, core=pin(PRODUCT/'runtime/Lokad.Onnx.dll'))


def prepare():
    assert not BASE.exists(); owner = previous_closed()
    product = read(PRODUCT/'closed.json'); assert product['passed']
    for name, wanted in product['files'].items(): assert pin(PRODUCT/name) == wanted, name
    for identity in product['identities']: monitor.terminal(identity)
    BASE.mkdir(); (BASE/'logs').mkdir(); (BASE/'output').mkdir(); (BASE/'runtime-source').mkdir()
    sources = []
    for relative in ['src/coreclr/inc/clrconfigvalues.h', 'src/coreclr/jit/jitconfigvalues.h']:
        url = 'https://raw.githubusercontent.com/dotnet/runtime/v10.0.8/'+relative
        path = BASE/'runtime-source'/Path(relative).name
        with urllib.request.urlopen(url, timeout=30) as response: data = response.read()
        assert b'EnableAVX512' in data and b'EnableAVX512F' not in data
        path.write_bytes(data); sources.append(dict(url=url, **pin(path)))
    save(BASE/'runtime-source/sources.json', sources)
    shutil.copytree(PRODUCT/'runtime', BASE/'runtime'); assert pin(BASE/'runtime/Lokad.Onnx.dll')['sha256'] == CORE
    for mode, previous, filename in [('raw', RAW/'source', 'Probe.cs'), ('layers', FAILED/'source', 'ModelProbe.cs')]:
        target = BASE/'source'/mode; target.mkdir(parents=True)
        for p in previous.iterdir():
            if p.is_file(): shutil.copy2(p, target/p.name)
        before = (target/filename).read_text(); assert before.count('DOTNET_EnableAVX512F') == 2
        after = before.replace('DOTNET_EnableAVX512F', 'DOTNET_EnableAVX512')
        (target/filename).write_text(after, encoding='utf8')
        (BASE/(mode+'-consumer.diff')).write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile='failed-flag-consumer', tofile='correct-runtime-flag')), encoding='utf8')
        project, = target.glob('*.csproj'); text = project.read_text(); assert text.count('../runtime/') == 2
        project.write_text(text.replace('../runtime/', '../../runtime/'), encoding='utf8')
    bridge = BASE/'source/bridge'; bridge.mkdir()
    code = (BRIDGE/'Program.cs').read_text(); assert code.count('args.Length != 3') == code.count('new[] { "GraphQualification.dll" }') == 1
    code = code.replace('args.Length != 3', 'args.Length != 4').replace('new[] { "GraphQualification.dll" }', 'new[] { args[3] }')
    (bridge/'Program.cs').write_text(code, encoding='utf8'); shutil.copy2(BRIDGE/'Bridge.csproj', bridge/'Bridge.csproj')
    inputs = {p.as_posix(): pin(p) for folder in [TOOLS, BASE/'source', BASE/'runtime', BASE/'runtime-source'] for p in folder.rglob('*') if p.is_file()}
    for p in [MONITOR, FAILED/'failure-closed.json', PRODUCT/'closed.json', RAW/'closed.json', BRIDGE/'Program.cs',
        BASE/'raw-consumer.diff', BASE/'layers-consumer.diff', ROOT/'tests/pyannote/blocked-spatial-raw-graphs/audit.py']:
        inputs[p.as_posix()] = pin(p)
    save(BASE/'local-inputs.json', dict(files=inputs, runtime_source=sources))
    own = monitor.psutil.Process(); state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    path = BASE/'local-controller.json'; flags = monitor.FLAGS+['-p:NuGetAudit=false']

    def run(name, arguments, numerical=False):
        monitor.worker(state, path, name, arguments, ROOT, [0], 12 if numerical else 8, 8, 900, not numerical, BASE/'output' if numerical else BASE/'source')
        print(name, 'passed', flush=True)

    try:
        for mode in ['raw', 'layers', 'bridge']:
            project, = (BASE/'source'/mode).glob('*.csproj')
            run(mode+'-restore', ['dotnet', 'restore', project, *flags, '--source', FEED, '--packages', BASE/'packages'])
            run(mode+'-build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'])
        for mode, assembly in [('raw', 'Lokad.Onnx.Backend.Tests.dll'), ('layers', 'LayerGraphs.dll')]:
            built = BASE/'source'/mode/'bin/Release/net10.0'; assert pin(built/'Lokad.Onnx.dll')['sha256'] == CORE
            run(mode+'-inventory', ['dotnet', bridge/'bin/Release/net10.0/Bridge.dll', FAILED/'payload/runtime', built, BASE/(mode+'-instructions.json'), assembly])
        save(BASE/'instruction-review.json', review())
        fixture = FAILED/'payload/fixtures'
        run('raw-local', ['dotnet', BASE/'source/raw/bin/Release/net10.0/Lokad.Onnx.Backend.Tests.dll', '256', BASE/'output/raw.json'], True)
        run('layers-local', ['dotnet', BASE/'source/layers/bin/Release/net10.0/LayerGraphs.dll', '256', fixture, BASE/'output/layers.json'], True)
        for mode, old in [('raw', RAW/'output/256.json'), ('layers', FAILED/'output/layers.json')]:
            result = read(BASE/'output'/(mode+'.json')); expected = read(old)
            assert result['passed'] and result['core'] == CORE and not result['flags']
            for key in (['observations', 'graph_cases', 'supplemental'] if mode == 'raw' else ['observations', 'graph_dispatch']):
                assert result[key] == expected[key], (mode, key)
        monitor.verify(inputs); previous_closed()
        payload = BASE/'payload'; payload.mkdir(); (payload/'tools').mkdir(); (payload/'fixtures').mkdir()
        shutil.copytree(BASE/'runtime', payload/'runtime')
        shutil.copy2(fixture/'result.json', payload/'fixtures/result.json')
        for name in ['remote.py', 'protocol.py']: shutil.copy2(TOOLS/name, payload/'tools'/name)
        for mode, name in [('raw', 'Lokad.Onnx.Backend.Tests'), ('layers', 'LayerGraphs')]:
            for suffix in ['dll', 'deps.json', 'runtimeconfig.json']:
                shutil.copy2(BASE/'source'/mode/'bin/Release/net10.0'/(name+'.'+suffix), payload/'runtime'/(name+'.'+suffix))
            shutil.copy2(BASE/'output'/(mode+'.json'), payload/('windows-'+mode+'.json'))
        shutil.copy2(ROOT/'.agent/m17-pyannote-blocked-product-20260922.md', payload/'prospective-plan.md')
        old = read(FAILED/'payload/payload.json'); external = dict(old['external'])
        for p in fixture.iterdir():
            if p.is_file(): external[REMOTE_FIXTURES+'/'+p.name] = pin(p)
        manifest = dict(passed=True, limits=LIMITS, previous_owner=owner, boot_time=1789634288.0,
            interpreter=old['interpreter'], external=external, fixture_directory=REMOTE_FIXTURES,
            core=pin(payload/'runtime/Lokad.Onnx.dll'), consumers=dict(raw=pin(payload/'runtime/Lokad.Onnx.Backend.Tests.dll'), layers=pin(payload/'runtime/LayerGraphs.dll')),
            jobs=['raw-256', 'raw-512', 'layers-256', 'layers-512'],
            files={p.relative_to(payload).as_posix(): pin(p) for p in payload.rglob('*') if p.is_file()},
            scope='Correct .NET 10 switch; actual normal-product raw/layer graphs in both AMD widths; no timing')
        save(payload/'payload.json', manifest)
        files = {p.relative_to(ROOT).as_posix(): pin(p) for p in [FAILED/'failure-closed.json', PRODUCT/'closed.json', RAW/'closed.json',
            BASE/'local-inputs.json', BASE/'instruction-review.json', BASE/'raw-instructions.json', BASE/'layers-instructions.json',
            BASE/'output/raw.json', BASE/'output/layers.json', *TOOLS.iterdir()] if p.is_file()}
        for p in TOOLS.glob('*.py'): ast.parse(p.read_text(), str(p))
        with tarfile.open(BASE/'payload.tar.gz', 'w:gz') as archive:
            for p in sorted(payload.rglob('*')):
                if p.is_file(): archive.add(p, arcname=p.relative_to(payload).as_posix(), recursive=False)
        save(BASE/'prepared.json', dict(passed=True, files=files, payload=pin(payload/'payload.json'), archive=pin(BASE/'payload.tar.gz')))
        print(json.dumps(dict(payload=pin(payload/'payload.json'), archive=pin(BASE/'payload.tar.gz'), files=len(manifest['files']), review=review())))
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); raise
    finally:
        state['complete'] = True; save(path, state)


if __name__ == '__main__': prepare()
