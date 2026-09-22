"""Correct two literal identity guards; prove every other consumer instruction equal."""
import importlib.util
import json
from pathlib import Path
import shutil
import traceback

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'artifacts/pyannote-combined-consumers-20260922'
OLD = ROOT / 'artifacts/pyannote-combined-amd-payload-20260922/payload'
FAILURE = ROOT / 'artifacts/pyannote-combined-amd-execution-20260922'
TOOLS = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('consumer_monitor', ROOT / 'tests/parakeet/packing-budgets/common.py')
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)
monitor.BASE = BASE
pin, read, save, verify, psutil = monitor.pin, monitor.read, monitor.save, monitor.verify, monitor.psutil


def main():
    assert not BASE.exists()
    assert pin(FAILURE / 'failure-closed.json')['sha256'] == 'afdda8b80915238c975c47f18e76e1ce7d99a032beeaa76de1ac5e780bdd601a'
    closed = read(FAILURE / 'failure-closed.json')
    for name, wanted in closed['files'].items():
        assert pin(FAILURE / name) == wanted
    BASE.mkdir()
    (BASE / 'logs').mkdir()
    files = {}
    for p in [Path(__file__).resolve(), ROOT / 'tests/parakeet/packing-budgets/common.py', FAILURE / 'failure-closed.json']:
        files[p.relative_to(ROOT).as_posix()] = pin(p)
    old_sha = read(OLD / 'manifests/production-pyannote.json')['data_sha256']
    projects = {}
    for role in ['portable', 'rows']:
        folder = BASE / role
        shutil.copytree(OLD / 'graph-consumer', folder)
        path = folder / 'Program.cs'
        before = path.read_text(encoding='utf8')
        sha = read(OLD / 'manifests' / (role + '-pyannote.json'))['data_sha256']
        assert before.count(old_sha) == 1 and old_sha != sha
        after = before.replace(old_sha, sha)
        assert after.replace(sha, old_sha) == before
        path.write_text(after, encoding='utf8')
        projects[role] = folder / 'GraphQualification.csproj'
    bridge = BASE / 'bridge'
    bridge.mkdir()
    source = (ROOT / 'tests/pyannote/combined-avx512/Inventory.cs.txt').read_text(encoding='utf8')
    old = 'foreach (string name in new[] { "Lokad.Onnx.dll", "Lokad.Onnx.Data.dll" })'
    assert source.count(old) == 1
    source = source.replace(old, 'foreach (string name in new[] { "GraphQualification.dll" })')
    (bridge / 'Program.cs').write_text(source, encoding='utf8')
    shutil.copy2(ROOT / 'artifacts/pyannote-portable-integration-20260922/bridge/Bridge.csproj', bridge / 'Bridge.csproj')
    projects['bridge'] = bridge / 'Bridge.csproj'
    for folder in [BASE / 'portable', BASE / 'rows', bridge]:
        for p in folder.iterdir():
            if p.is_file():
                files[p.relative_to(ROOT).as_posix()] = pin(p)
    save(BASE / 'inputs.json', dict(passed=True, files=files))
    own = psutil.Process()
    state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    save(BASE / 'processes.json', state)
    flags = monitor.FLAGS + ['-p:NuGetAudit=false']

    def run(name, args, output):
        monitor.worker(state, BASE / 'processes.json', name, args, BASE, [0], 8, 8, 900, True, output)
        print(name, 'passed', flush=True)

    try:
        for name, project in projects.items():
            run(name + '-restore', ['dotnet', 'restore', project, *flags, '--source', OLD / 'nuget-feed',
                '--packages', BASE / 'packages'], None)
            extra = [] if name == 'bridge' else ['-p:FrozenProductDirectory=' + str(OLD / 'runtimes' / name)]
            run(name + '-build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore',
                '--disable-build-servers', *extra], project.parent)
        reports = {}
        for role in ['portable', 'rows']:
            runtime = projects[role].parent / 'bin/Release/net10.0'
            for name in ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll']:
                assert pin(runtime / name) == pin(OLD / 'runtimes' / role / name)
            run(role + '-instructions', ['dotnet', bridge / 'bin/Release/net10.0/Bridge.dll',
                OLD / 'runtimes' / role, runtime, BASE / (role + '-instructions.json')], bridge)
            inventory = read(BASE / (role + '-instructions.json'))
            assert inventory['inventory_complete'] and len(inventory['observations']) == 1
            row = inventory['observations'][0]
            assert row['public_surface_equal'] and not row['added'] and not row['removed'] and len(row['differences']) == 1
            method = row['differences'][0]
            assert method.startswith('Program::<Main>$::')
            before, after = row['normalized_methods'][method], row['candidate_methods'][method]
            data_sha = read(OLD / 'manifests' / (role + '-pyannote.json'))['data_sha256']
            assert before.count(old_sha) == 1 and before.replace(old_sha, data_sha) == after
            reports[role] = dict(passed=True, methods=row['methods'], changed_method=method,
                sole_change='Data SHA256 literal', before=old_sha, after=data_sha,
                consumer=pin(runtime / 'GraphQualification.dll'), inventory=pin(BASE / (role + '-instructions.json')))
        verify(files)
        for p in BASE.rglob('*'):
            if p.is_file() and not {'obj', 'packages', 'logs'}.intersection(p.relative_to(BASE).parts):
                files[p.relative_to(ROOT).as_posix()] = pin(p)
        # processes.json will receive its terminal state below; bind it during closure.
        files.pop((BASE / 'processes.json').relative_to(ROOT).as_posix(), None)
        save(BASE / 'prepared.json', dict(passed=True, files=files, reports=reports,
            product_binaries_unchanged=True, inference_assertions_unchanged=True))
        state['code'] = 0
        print(json.dumps(reports), flush=True)
    except BaseException:
        state.update(code=1, error=traceback.format_exc())
        raise
    finally:
        state['complete'] = True
        save(BASE / 'processes.json', state)


if __name__ == '__main__':
    main()
