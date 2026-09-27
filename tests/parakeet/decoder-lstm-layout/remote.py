"""Use the existing serial supervisor unchanged, with build/contract callbacks."""
import importlib.util
from pathlib import Path
import shutil
from protocol import pin, read, save
from commands import command_for as commands
from checks import compiled, contracts

TOOLS = Path(__file__).resolve().parent
BASE = TOOLS.parent
source = TOOLS/'remote_base.py'
if not source.exists(): source = TOOLS.parent/'dispatch-events-amd/remote.py'
loader = importlib.util.spec_from_file_location('lstm_layout_supervisor', source)
worker = importlib.util.module_from_spec(loader); loader.loader.exec_module(worker)
worker.BASE = BASE
idle, live = worker.idle, worker.live


def command_for(name, spec): return commands(BASE, name, spec)


def after(name, spec, row):
    if name == 'sdk-version': assert (BASE/'logs/sdk-version.stdout').read_text().strip() == '10.0.204'
    if name == 'backend-build':
        folder = BASE/'source/tests/Lokad.Onnx.Backend.Tests/bin/Release/net10.0'
        target = BASE/'runtimes/candidate'; target.mkdir()
        for prior in (BASE/'runtimes/current').glob('*.dll'):
            filename = prior.name
            source = folder/filename if filename in ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll'] else prior
            shutil.copy2(source, target/filename)
        files = {p.relative_to(BASE).as_posix(): pin(p) for p in folder.rglob('*') if p.is_file()}
        files.update({p.relative_to(BASE).as_posix(): pin(p) for p in (BASE/'runtimes').rglob('*') if p.is_file()})
        product = {name: pin(target/name) for name in ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll']}
        save(BASE/'built.json', dict(passed=True, files=files, candidate=product,
            products=dict(current=spec['current_product']['Lokad.Onnx.dll'], candidate=product['Lokad.Onnx.dll']),
            consumer=pin(folder/'Lokad.Onnx.Backend.Tests.dll')))
    if name == 'inventory':
        save(BASE/'inventory/review.json', compiled(read(BASE/'inventory/instructions.json'),
            spec['current_product'], read(BASE/'built.json')['candidate']))
    if name.startswith('contracts-'):
        save(BASE/name/'review.json', contracts(BASE/name, spec, read(BASE/'built.json'), row))


worker.command_for, worker.after = command_for, after
main = worker.main
if __name__ == '__main__': raise SystemExit(main())
