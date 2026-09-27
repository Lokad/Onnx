"""Reuse the EventPipe supervisor, changing only this diagnostic's callbacks."""
import importlib.util
from pathlib import Path
import shutil
from protocol import JOBS, pin, read, save
from checks import result

TOOLS = Path(__file__).resolve().parent; BASE = TOOLS.parent
source = TOOLS/'remote_base.py'
if not source.exists(): source = TOOLS.parent/'dispatch-events-amd/remote.py'
loader = importlib.util.spec_from_file_location('lstm_runtime_supervisor', source)
worker = importlib.util.module_from_spec(loader); loader.loader.exec_module(worker)
worker.BASE = BASE
DOTNET, FLAGS = worker.DOTNET, worker.FLAGS
idle, live = worker.idle, worker.live


def command_for(name, spec):
    assert name in JOBS
    if name == 'sdk-version': return [DOTNET, '--version'], True, 2
    if name == 'tracer-version': return [DOTNET, BASE/'tracer/dotnet-trace.dll', '--version'], False, 0
    if name.startswith('observer-'):
        action = name.split('-')[1]
        command = [DOTNET, action, BASE/'source/Timing.csproj', *FLAGS]
        command += ['--source', spec['feed'], '--packages', BASE/'packages'] if action == 'restore' else ['-c', 'Release', '--no-restore', '--disable-build-servers']
        return command, True, 2
    if name == 'trace-capture':
        return [DOTNET, BASE/'runtime/ParakeetRecurrenceTiming.dll', BASE, 'selectedfallback', '512', name, BASE/name/'output'], False, 2
    if name == 'trace-export':
        return [DOTNET, BASE/'export-runtime/DispatchEventsExport.dll', BASE/'trace-capture/capture.nettrace', BASE/name/'events'], False, 0
    assert name == 'trace-stacks'
    return [DOTNET, BASE/'tracer/dotnet-trace.dll', 'convert', BASE/'trace-capture/capture.nettrace', '--format', 'Speedscope', '--output', BASE/name/'speedscope'], False, 0


def after(name, spec, row):
    if name == 'sdk-version': assert (BASE/'logs/sdk-version.stdout').read_text().strip() == '10.0.204'
    if name == 'observer-build':
        folder = BASE/'source/bin/Release/net10.0'
        for filename, wanted in spec['product'].items(): assert pin(folder/filename) == wanted, filename
        files = {}
        for suffix in ['dll', 'deps.json', 'runtimeconfig.json']:
            source = folder/('ParakeetRecurrenceTiming.'+suffix); target = BASE/'runtime'/source.name
            assert not target.exists(); shutil.copy2(source, target); files[target.relative_to(BASE).as_posix()] = pin(target)
        save(BASE/'built.json', dict(passed=True, files=files, consumer=pin(BASE/'runtime/ParakeetRecurrenceTiming.dll')))
    if name == 'trace-capture': save(BASE/name/'review.json', result(BASE, spec, row, read(BASE/'built.json')))
    if name == 'trace-export':
        summary = read(BASE/name/'events/summary.json')
        assert summary['complete'] and summary['lost'] == 0 and summary['clr_events'] > 0
        assert summary['protocol'] == 'all-event-records-v1'
        assert summary['input_sha256'] == pin(BASE/'trace-capture/capture.nettrace')['sha256']
    if name == 'trace-stacks': assert len(list((BASE/name).glob('*.speedscope.json'))) == 1


worker.command_for, worker.after = command_for, after
main = worker.main
if __name__ == '__main__': raise SystemExit(main())
