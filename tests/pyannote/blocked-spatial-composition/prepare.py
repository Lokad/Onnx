"""Build an isolated normal-source composition against the admitted component."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import traceback
from transform import apply

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/pyannote-blocked-spatial-composition-20260922'
COMPONENT = ROOT/'artifacts/pyannote-vector-input-layout-20260922'
AMD = ROOT/'artifacts/pyannote-vector-input-layout-amd-20260922'
CONTROL = ROOT/'artifacts/pyannote-single-panel-composition-20260922/runtime'
FEED = ROOT/'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
MONITOR = ROOT/'tests/parakeet/packing-budgets/common.py'
spec = importlib.util.spec_from_file_location('blocked_product_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec); spec.loader.exec_module(monitor); monitor.BASE = BASE
pin, read, save, verify, terminal = monitor.pin, monitor.read, monitor.save, monitor.verify, monitor.terminal


def review():
    inventory = read(BASE/'instructions.json'); assert inventory['inventory_complete']
    observations = []
    allowed = {
        'Lokad.Onnx.ComputationalGraph': {'.ctor', 'InvalidatePreparation', 'RunCoreInner', 'RefreshLifetimeAnalysis'},
        'Lokad.Onnx.GraphExecution': {'.ctor'},
        'Lokad.Onnx.GraphPacking': {'PackMatMulWeights'},
        'Lokad.Onnx.Tensor`1[T]': {'Conv2DFloatCore'},
        'Lokad.Onnx.TensorExecutionOptions': {'.ctor', 'GetHashCode', 'Equals', 'PrintMembers', 'ToString'}
    }
    for row in inventory['observations']:
        assert row['public_surface_equal'] and not row['removed']
        assert row['before_sha256'] == pin(CONTROL/row['assembly'])['sha256']
        assert row['after_sha256'] == pin(BASE/'runtime'/row['assembly'])['sha256']
        if row['assembly'] == 'Lokad.Onnx.Data.dll':
            assert row['methods'] == row['unchanged_methods'] == 697 and not row['added'] and not row['differences']
        else:
            assert row['methods'] == 3113
            for key in row['differences']:
                owner, name, _ = key.split('::'); assert name in allowed.get(owner, set()), key
            for owner, name in [('Lokad.Onnx.Tensor`1[T]', 'Conv2DFloatCore'), ('Lokad.Onnx.GraphPacking', 'PackMatMulWeights'),
                ('Lokad.Onnx.ComputationalGraph', 'InvalidatePreparation'), ('Lokad.Onnx.ComputationalGraph', 'RunCoreInner'),
                ('Lokad.Onnx.ComputationalGraph', 'RefreshLifetimeAnalysis')]:
                assert any(k.startswith(owner+'::'+name+'::') for k in row['differences'])
            for key in row['added']:
                owner, name, _ = key.split('::')
                assert owner.startswith(('Lokad.Onnx.ConvBlockedSpatial', 'Lokad.Onnx.GraphConvPacking', 'Lokad.Onnx.PackedConvWeight')) or (
                    owner == 'Lokad.Onnx.Tensor`1[T]' and name in ['PlanConvBlockedScratch', 'TryConvBlockedSpatial']) or (
                    owner == 'Lokad.Onnx.TensorExecutionOptions' and name in ['get_PackedConvWeights', 'set_PackedConvWeights']), key
            equal = []
            for key, body in inventory['component_methods'].items():
                if not key.startswith('BlockedSpatial::') or key.split('::')[1] in ['Execute', 'Run']: continue
                normalized_key = key.replace('BlockedSpatial::', 'Lokad.Onnx.ConvBlockedSpatial::')
                normalized_body = body.replace('BlockedSpatial::', 'Lokad.Onnx.ConvBlockedSpatial::')
                assert row['candidate_methods'][normalized_key] == normalized_body, key
                equal.append(key)
            assert len(equal) >= 10
            observations.append(dict(assembly=row['assembly'], methods=row['methods'], unchanged=row['unchanged_methods'],
                changed=row['differences'], added=row['added'], component_methods_equal=equal))
    return dict(passed=True, observations=observations, data_methods_unchanged=697,
        public_surface_equal=True, instructions=pin(BASE/'instructions.json'))


def main():
    assert not BASE.exists()
    for folder, sha in [(COMPONENT, '8dae29462e8d0694031b41e0db31387da550b0796c2fc386ebecf2f00e21a978'),
        (AMD, 'b244e60c8d1a04ebc3e876e44d3b3d41f7a0667f9d71eca12913ff00099e9da2')]:
        assert pin(folder/'closed.json')['sha256'] == sha
        proof = read(folder/'closed.json'); assert proof['passed']
        for name, wanted in proof['files'].items(): assert pin(folder/name) == wanted, name
    assert read(AMD/'closed.json')['admitted'] and read(AMD/'collected/collection.json')['terminal']
    for identity in read(COMPONENT/'closed.json')['identities']: terminal(identity)
    assert pin(CONTROL/'Lokad.Onnx.dll')['sha256'] == '1279b4b662241db2404fa4875eae20eaa924f15677a85655f99b8f81cd24b309'
    assert pin(CONTROL/'Lokad.Onnx.Data.dll')['sha256'] == '4e602d9f6a35a51277d6deb9d75779d84cecf0a3a433d1b4eb70b0006462cca4'
    head = subprocess.check_output(['git', 'rev-parse', 'a0cc7741'], cwd=ROOT, text=True).strip()
    BASE.mkdir(); (BASE/'logs').mkdir(); (BASE/'test-results').mkdir()
    own = monitor.psutil.Process(); state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    state_file = BASE/'preparation.json'; save(state_file, state)
    try:
        archive = BASE/'source.tar'
        subprocess.run(['git', 'archive', '--format=tar', '--output', str(archive), head], cwd=ROOT, check=True)
        source = BASE/'source'; source.mkdir()
        with tarfile.open(archive) as tar: tar.extractall(source, filter='data')
        (BASE/'candidate.patch').write_text(apply(source, COMPONENT/'source', TOOLS), encoding='utf8')
        shutil.copy2(ROOT/'.agent/m17-pyannote-blocked-product-20260922.md', BASE/'prospective-plan.md')
        bridge = BASE/'bridge'; bridge.mkdir()
        code = (ROOT/'tests/pyannote/combined-avx512/Inventory.cs.txt').read_text()
        assert code.count('args.Length != 3') == code.count('observations }, new JsonSerializerOptions') == 1
        code = code.replace('args.Length != 3', 'args.Length != 4').replace('observations }, new JsonSerializerOptions',
            'observations, component_methods = Inspect(Load(Path.GetFullPath(args[3]), "VectorInputProbe.dll", "component")) }, new JsonSerializerOptions')
        (bridge/'Program.cs').write_text(code, encoding='utf8')
        shutil.copy2(ROOT/'artifacts/pyannote-portable-integration-20260922/bridge/Bridge.csproj', bridge/'Bridge.csproj')
        files = {p.as_posix(): pin(p) for folder in [source, bridge, TOOLS] for p in folder.rglob('*') if p.is_file()}
        for p in [archive, MONITOR, BASE/'candidate.patch', BASE/'prospective-plan.md', COMPONENT/'closed.json', AMD/'closed.json',
            COMPONENT/'source/bin/Release/net10.0/VectorInputProbe.dll', CONTROL/'Lokad.Onnx.dll', CONTROL/'Lokad.Onnx.Data.dll']:
            files[p.as_posix()] = pin(p)
        save(BASE/'inputs.json', dict(files=files, source_commit=head, production_changed=False))
        flags = monitor.FLAGS+['-p:NuGetAudit=false']
        def run(name, command, inference=False):
            monitor.worker(state, state_file, name, command, source, [0], 12 if inference else 8, 8, 900, True, None)
            print(name, 'passed', flush=True)
        for name, project in [('cli', source/'src/Lokad.Onnx.CLI/Lokad.Onnx.CLI.csproj'),
            ('backend', source/'tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj'),
            ('tensors', source/'tests/Lokad.Onnx.Tensors.Tests/Lokad.Onnx.Tensors.Tests.csproj'), ('bridge', bridge/'Bridge.csproj')]:
            run(name+'-restore', ['dotnet', 'restore', project, *flags, '--source', FEED, '--packages', BASE/'packages'])
            run(name+'-build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'])
        shutil.copytree(source/'src/Lokad.Onnx.CLI/bin/Release/net10.0', BASE/'runtime')
        for name in ['Backend', 'Tensors']:
            assert pin(source/f'tests/Lokad.Onnx.{name}.Tests/bin/Release/net10.0/Lokad.Onnx.dll') == pin(BASE/'runtime/Lokad.Onnx.dll')
        run('inventory', ['dotnet', bridge/'bin/Release/net10.0/Bridge.dll', CONTROL, BASE/'runtime', BASE/'instructions.json', COMPONENT/'source/bin/Release/net10.0'])
        save(BASE/'instruction-review.json', review())
        for mode in ['normal', 'disabled']:
            previous = monitor.clean_env
            if mode == 'disabled': monitor.clean_env = lambda: previous() | {'DOTNET_EnableHWIntrinsic': '0'}
            try:
                run('focused-'+mode, ['dotnet', 'test', source/'tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj',
                    '-c', 'Release', *flags, '--no-build', '--no-restore', '--filter', 'FullyQualifiedName~ConvBlockedSpatialTests',
                    '--logger', 'trx;LogFileName=focused-'+mode+'.trx', '--results-directory', BASE/'test-results'], True)
            finally: monitor.clean_env = previous
        verify(files)
        for folder in [source, bridge, BASE/'runtime']:
            files.update({p.as_posix(): pin(p) for p in folder.rglob('*') if p.is_file() and not {'obj', 'packages'}.intersection(p.relative_to(folder).parts)})
        save(BASE/'prepared.json', dict(passed=True, files=files, core=pin(BASE/'runtime/Lokad.Onnx.dll'),
            data=pin(BASE/'runtime/Lokad.Onnx.Data.dll'), source_commit=head, production_changed=False, models_qualified=False, performance_qualified=False))
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); raise
    finally:
        state['complete'] = True; save(state_file, state)


if __name__ == '__main__': main()
