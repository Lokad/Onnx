"""Reuse the original serial supervisor and collector with eight bounded jobs."""
import importlib.util
from pathlib import Path
import shutil
from protocol import JOBS, LIMITS, PROVIDERS, pin, read, save
from checks import result

TOOLS = Path(__file__).resolve().parent
BASE = TOOLS.parent
SOURCE = TOOLS/'remote_base.py'
if not SOURCE.exists(): SOURCE = TOOLS.parent/'dispatch-events-amd/remote.py'
loader = importlib.util.spec_from_file_location('decoder_inherited_supervisor', SOURCE)
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
        command = [DOTNET, action, BASE/'source/observer/Observer.csproj', *FLAGS]
        command += ['--source', spec['feed'], '--packages', BASE/'packages'] if action == 'restore' else ['-c', 'Release', '--no-restore', '--disable-build-servers']
        return command, True, 2
    if name in ['control-run', 'trace-capture']:
        return [DOTNET, BASE/'runtime/ParakeetDecoderObservation.dll', BASE/'observation.json',
                'control' if name == 'control-run' else 'trace', BASE/name], False, 2
    if name == 'trace-export':
        return [DOTNET, BASE/'export-runtime/DispatchEventsExport.dll', BASE/'trace-capture/capture.nettrace', BASE/name/'events'], False, 0
    assert name == 'trace-stacks'
    return [DOTNET, BASE/'tracer/dotnet-trace.dll', 'convert', BASE/'trace-capture/capture.nettrace',
            '--format', 'Speedscope', '--output', BASE/name/'speedscope'], False, 0


def after(name, spec, row):
    if name == 'sdk-version':
        assert (BASE/'logs/sdk-version.stdout').read_text().strip().endswith('10.0.204')
    elif name == 'observer-build':
        folder = BASE/'source/observer/bin/Release/net10.0'
        assert pin(folder/'Lokad.Onnx.dll') == spec['product']['Lokad.Onnx.dll']
        assert pin(folder/'Google.Protobuf.dll') == pin(BASE/'runtime/Google.Protobuf.dll')
        built = dict(passed=True, files={})
        for suffix in ['dll', 'deps.json', 'runtimeconfig.json']:
            source = folder/('ParakeetDecoderObservation.'+suffix)
            target = BASE/'runtime'/source.name
            assert not target.exists(); shutil.copy2(source, target)
            built['files'][target.relative_to(BASE).as_posix()] = pin(target)
        built['consumer'] = pin(BASE/'runtime/ParakeetDecoderObservation.dll')
        save(BASE/'built.json', built)
    elif name in ['control-run', 'trace-capture']:
        save(BASE/name/'review.json', result(BASE, name, spec, row['processes']['worker'], read(BASE/'built.json')))
    elif name == 'trace-export':
        summary = read(BASE/name/'events/summary.json')
        assert summary['complete'] and summary['lost'] == 0 and summary['clr_events'] > 0
        assert summary['protocol'] == 'all-event-records-v1'
        assert summary['input_sha256'] == pin(BASE/'trace-capture/capture.nettrace')['sha256']
    elif name == 'trace-stacks':
        assert len(list((BASE/name).glob('*.speedscope.json'))) == 1


worker.command_for, worker.after = command_for, after
main = worker.main

if __name__ == '__main__': raise SystemExit(main())
