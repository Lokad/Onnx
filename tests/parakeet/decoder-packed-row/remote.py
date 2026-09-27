"""Run the existing resource supervisor with one build and fixed contract modes."""
import importlib.util
from pathlib import Path
import shutil
from protocol import JOBS, LIMITS, PROVIDERS, pin, read, save
from commands import command_for as commands
from compiled import review

TOOLS = Path(__file__).resolve().parent
BASE = TOOLS.parent
SOURCE = TOOLS/'remote_base.py'
if not SOURCE.exists(): SOURCE = TOOLS.parent/'dispatch-events-amd/remote.py'
loader = importlib.util.spec_from_file_location('packed_row_supervisor', SOURCE)
worker = importlib.util.module_from_spec(loader); loader.loader.exec_module(worker)
worker.BASE = BASE
idle, live = worker.idle, worker.live


def command_for(name, spec): return commands(BASE, name, spec)


def after(name, spec, row):
    if name == 'sdk-version':
        assert (BASE/'logs/sdk-version.stdout').read_text().strip() == '10.0.204'
    elif name == 'core-build':
        source = BASE/'source/src/Lokad.Onnx/bin/Release/net10.0/Lokad.Onnx.dll'
        target = BASE/'runtimes/candidate/Lokad.Onnx.dll'
        assert not target.exists(); shutil.copy2(source, target)
    elif name == 'contracts-build':
        folder = BASE/'source/contracts/bin/Release/net10.0'
        assert pin(folder/'Lokad.Onnx.dll') == spec['current_product']['Lokad.Onnx.dll']
        for role in ['current', 'candidate']:
            for suffix in ['dll', 'deps.json', 'runtimeconfig.json']:
                source = folder/('Lokad.Onnx.Backend.Tests.' + suffix)
                target = BASE/'runtimes'/role/source.name
                assert not target.exists(); shutil.copy2(source, target)
        files = {p.relative_to(BASE).as_posix(): pin(p) for p in (BASE/'runtimes').rglob('*') if p.is_file()}
        save(BASE/'built.json', dict(passed=True, files=files,
            products={role: files['runtimes/'+role+'/Lokad.Onnx.dll'] for role in ['current', 'candidate']},
            consumer=files['runtimes/current/Lokad.Onnx.Backend.Tests.dll']))
    elif name == 'inventory':
        save(BASE/'inventory/review.json', review(read(BASE/'inventory/instructions.json'), read(BASE/'built.json'), spec['current_product']))
    elif name.startswith(('current-', 'candidate-')):
        role, mode = name.split('-'); value = read(BASE/name/'result.json')
        assert value['passed'] and value['role'] == role and value['mode'] == mode
        assert value['pid'] == row['processes']['worker']['pid']
        built = read(BASE/'built.json')
        assert value['core_sha256'] == built['products'][role]['sha256']
        assert value['consumer_sha256'] == built['consumer']['sha256']
        assert len(value['public_cases']) == 45
        assert len(value['raw_cases']) == (81 if role == 'candidate' and mode != 'scalar' else 0)
        assert not value['performance_admitted']


worker.command_for, worker.after = command_for, after
main = worker.main
if __name__ == '__main__': raise SystemExit(main())
