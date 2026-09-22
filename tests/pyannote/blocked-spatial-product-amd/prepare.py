"""Freeze actual normal-product consumers, fixtures and a four-worker AMD protocol."""
import ast
import difflib
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
import traceback
from protocol import LIMITS, pin, read, save

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/pyannote-blocked-spatial-product-amd-20260922'
PRIOR = ROOT/'artifacts/pyannote-vector-input-layout-amd-20260922'
RAW = ROOT/'artifacts/pyannote-blocked-spatial-raw-graphs-20260922'
LAYERS = ROOT/'artifacts/pyannote-blocked-spatial-layer-graphs-20260922'
PRODUCT = ROOT/'artifacts/pyannote-blocked-spatial-composition-v3-20260922'
FIXTURES = ROOT/'artifacts/pyannote-blocked-spatial-fixtures-20260922'
CORE = '3c2f16b08856426d3dfeff07f1638dd76cee7f06b65bbee230e8e0789679206f'
FEED = ROOT/'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
MONITOR = ROOT/'tests/parakeet/packing-budgets/common.py'
spec = importlib.util.spec_from_file_location('product_amd_local_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec); spec.loader.exec_module(monitor); monitor.BASE = BASE


def closed(folder, sha):
    assert pin(folder/'closed.json')['sha256'] == sha
    proof = read(folder/'closed.json'); assert proof['passed']
    for name, wanted in proof['files'].items(): assert pin(folder/name) == wanted, name
    for identity in proof.get('identities', []): monitor.terminal(identity)
    return proof


def previous_closed():
    proof = closed(PRIOR, 'b244e60c8d1a04ebc3e876e44d3b3d41f7a0667f9d71eca12913ff00099e9da2'); assert proof['admitted']
    receipt = read(PRIOR/'collected/collection.json'); assert receipt['terminal'] and receipt['input_error'] is None
    return read(PRIOR/'deployment.json')


def prepare():
    assert not BASE.exists(); owner = previous_closed()
    closed(RAW, '8c28301080a91b9760b8e9239d7e25a2cd8471c3c543a6ba596b5b2be1d067c9')
    closed(LAYERS, '3edddec577c52d31d1ec97b952e3ddbd894b26dbe902835cccde958bf59209b4')
    closed(PRODUCT, 'e7a9a30d88ef2a425c0c51b007e2b7d89428d54e50ef7dc4e446e191f95719e3')
    closed(FIXTURES, '27ac5ab76c21828d18e59478f6010a4139df0530b349ba5863e5f771dc97422e')
    assert pin(PRODUCT/'runtime/Lokad.Onnx.dll')['sha256'] == CORE
    BASE.mkdir(); (BASE/'logs').mkdir(); source = BASE/'source'; source.mkdir(); (BASE/'output').mkdir()
    shutil.copytree(PRODUCT/'runtime', BASE/'runtime')
    for p in (LAYERS/'source').iterdir():
        if p.is_file(): shutil.copy2(p, source/p.name)
    before = (source/'ModelProbe.cs').read_text(); old = '91614ff625bbf46cadfeff3bab198d4aa7b707e508d93a3c043da41d716f47bf'
    assert before.count(old) == 1; after = before.replace(old, CORE)
    (source/'ModelProbe.cs').write_text(after, encoding='utf8')
    (BASE/'layer-identity.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
        fromfile='qualified-layer-consumer', tofile='final-normal-core-identity')), encoding='utf8')
    inputs = {p.as_posix(): pin(p) for folder in [TOOLS, source, BASE/'runtime'] for p in folder.rglob('*') if p.is_file()}
    for p in [MONITOR, RAW/'closed.json', LAYERS/'closed.json', PRODUCT/'closed.json', FIXTURES/'closed.json',
              BASE/'layer-identity.diff', ROOT/'tests/pyannote/blocked-spatial-raw-graphs/audit.py']:
        inputs[p.as_posix()] = pin(p)
    save(BASE/'local-inputs.json', dict(files=inputs))
    own = monitor.psutil.Process(); state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    path = BASE/'local-controller.json'; flags = monitor.FLAGS+['-p:NuGetAudit=false']; project = source/'LayerGraphs.csproj'
    try:
        monitor.worker(state, path, 'restore', ['dotnet', 'restore', project, *flags, '--source', FEED, '--packages', BASE/'packages'], ROOT, [0], 8, 8, 900, True, source)
        monitor.worker(state, path, 'build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'], ROOT, [0], 8, 8, 900, True, source)
        built = source/'bin/Release/net10.0'; assert pin(built/'Lokad.Onnx.dll')['sha256'] == CORE
        monitor.worker(state, path, 'layers-local', ['dotnet', built/'LayerGraphs.dll', '256', FIXTURES/'output', BASE/'output/layers.json'], ROOT, [0], 12, 8, 900, False, BASE/'output')
        result = read(BASE/'output/layers.json'); expected = read(LAYERS/'output/256.json')
        assert result['passed'] and result['core'] == CORE
        for key in ['observations', 'graph_dispatch', 'values', 'maximum', 'layer_graphs', 'graph_retained_bytes']:
            assert result[key] == expected[key], key
        monitor.verify(inputs)
        payload = BASE/'payload'; payload.mkdir(); (payload/'tools').mkdir()
        shutil.copytree(BASE/'runtime', payload/'runtime'); shutil.copytree(FIXTURES/'output', payload/'fixtures')
        for name in ['remote.py', 'protocol.py']: shutil.copy2(TOOLS/name, payload/'tools'/name)
        for name, folder in [('LayerGraphs', built), ('Lokad.Onnx.Backend.Tests', RAW/'source/bin/Release/net10.0')]:
            for suffix in ['dll', 'deps.json', 'runtimeconfig.json']: shutil.copy2(folder/(name+'.'+suffix), payload/'runtime'/(name+'.'+suffix))
        for name, folder in [('layers', source), ('raw', RAW/'source')]:
            target = payload/'source'/name; target.mkdir(parents=True)
            for p in folder.iterdir():
                if p.is_file(): shutil.copy2(p, target/p.name)
        shutil.copy2(RAW/'output/256.json', payload/'windows-raw.json')
        shutil.copy2(BASE/'output/layers.json', payload/'windows-layers.json')
        shutil.copy2(ROOT/'.agent/m17-pyannote-blocked-product-20260922.md', payload/'prospective-plan.md')
        previous = read(PRIOR/'payload/payload.json')
        manifest = dict(passed=True, limits=LIMITS, previous_owner=owner, boot_time=1789634288.0,
            interpreter=previous['interpreter'], external=previous['external'], core=pin(payload/'runtime/Lokad.Onnx.dll'),
            consumers=dict(raw=pin(payload/'runtime/Lokad.Onnx.Backend.Tests.dll'), layers=pin(payload/'runtime/LayerGraphs.dll')),
            jobs=['raw-256', 'raw-512', 'layers-256', 'layers-512'],
            files={p.relative_to(payload).as_posix(): pin(p) for p in payload.rglob('*') if p.is_file()},
            scope='Actual normal-product raw and captured-layer graphs in both AMD widths; no performance measurement')
        save(payload/'payload.json', manifest)
        files = {p.relative_to(ROOT).as_posix(): pin(p) for p in [RAW/'closed.json', LAYERS/'closed.json', PRODUCT/'closed.json',
            FIXTURES/'closed.json', PRIOR/'closed.json', BASE/'local-inputs.json', BASE/'layer-identity.diff', BASE/'output/layers.json', *TOOLS.iterdir()] if p.is_file()}
        for p in TOOLS.glob('*.py'): ast.parse(p.read_text(), str(p))
        with tarfile.open(BASE/'payload.tar.gz', 'w:gz') as archive:
            for p in sorted(payload.rglob('*')):
                if p.is_file(): archive.add(p, arcname=p.relative_to(payload).as_posix(), recursive=False)
        save(BASE/'prepared.json', dict(passed=True, files=files, payload=pin(payload/'payload.json'), archive=pin(BASE/'payload.tar.gz')))
        print(json.dumps(dict(payload=pin(payload/'payload.json'), archive=pin(BASE/'payload.tar.gz'), files=len(manifest['files']))))
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); raise
    finally:
        state['complete'] = True; save(path, state)


if __name__ == '__main__': prepare()
