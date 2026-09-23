"""Adapt the retained bounded event supervisor; no scoring or runtime flags."""
import shutil
import remote_base as supervisor
from remote_base import BASE, DOTNET, FLAGS, read, pin, save, live, idle
from protocol import DISASM
from checks import rows, codegen, codegen_bodies


def command_for(name, spec):
    if name == 'sdk-version': return [DOTNET, '--version'], True, 2
    if name == 'tracer-version': return [DOTNET, BASE / 'tracer/dotnet-trace.dll', '--version'], False, 0
    if name.startswith('producer-'):
        action = name.split('-')[1]
        command = [DOTNET, action, BASE / 'source/consumer/Producer.csproj', *FLAGS]
        command += ['--source', spec['feed'], '--packages', BASE / 'packages'] if action == 'restore' else ['-c', 'Release', '--no-restore', '--disable-build-servers']
        return command, True, 2
    role, action = name.split('-'); assert role in ['current', 'candidate']
    args = [BASE, role, ['current', 'candidate'].index(role), BASE / name / 'result.json']
    if action == 'capture': return [DOTNET, BASE / 'runtimes' / role / 'ParakeetDispatchEvents.dll', *args], False, 2
    if action == 'codegen': return ['/usr/bin/env', 'DOTNET_JitDisasm=' + DISASM, DOTNET, BASE / 'runtimes' / role / 'ParakeetIsolatedCodegen.dll', *args], False, 2
    assert action == 'export'
    return [DOTNET, BASE / 'export-runtime/DispatchEventsExport.dll', BASE / (role + '-capture/capture.nettrace'), BASE / name / 'events'], False, 0


def after(name, spec, row):
    if name == 'sdk-version': assert (BASE / 'logs/sdk-version.stdout').read_text().strip().endswith('10.0.204')
    if name == 'producer-build':
        built = dict(passed=True, files={}, exporter=spec['exporter'], capture_consumer=spec['capture_consumer'])
        folder = BASE / 'source/consumer/bin/Release/net10.0'
        for role in ['current', 'candidate']:
            for suffix in ['dll', 'deps.json', 'runtimeconfig.json']:
                source = folder / ('ParakeetIsolatedCodegen.' + suffix)
                target = BASE / 'runtimes' / role / source.name
                assert not target.exists(); shutil.copy2(source, target)
                built['files'][target.relative_to(BASE).as_posix()] = pin(target)
        built['consumer'] = pin(folder / 'ParakeetIsolatedCodegen.dll'); save(BASE / 'built.json', built)
    if name.endswith(('-capture', '-codegen')):
        role = name.split('-')[0]; value = read(BASE / name / 'result.json')
        rows(value, read(BASE / 'evidence/capture.json'))
        assert value['pid'] == row['processes']['worker']['pid'] and value['runtime'] == '10.0.8'
        assert value['role'] == role and value['sequence'] == ['current', 'candidate'].index(role)
        assert value['core_sha256'] == spec['products'][role]['Lokad.Onnx.dll']['sha256']
        if name.endswith('-capture'):
            ready = read(BASE / name / 'ready.json'); enabled = read(BASE / name / 'collector-enabled.json')
            assert value['pid'] == ready['pid'] == enabled['pid'] and value['nativeThread'] == ready['native_thread']
            assert ready['counter'] < enabled['counter'] < value['rows'][0]['clocks'][0]['marker']
            assert value['protocol'] == 'parakeet-dispatch-events-v1' and value['flags'] == {}
            assert value['assembly'] == spec['capture_consumer']['sha256']
            assert (BASE / name / 'capture.nettrace').stat().st_size > 0
        else:
            assert value['assembly'] == read(BASE / 'built.json')['consumer']['sha256']
            codegen(value, read(BASE / 'evidence/capture.json'))
            save(BASE / name / 'bodies.json', codegen_bodies((BASE / 'logs' / (name + '.stdout')).read_text(), role))
    if name.endswith('-export'):
        value = read(BASE / name / 'events/summary.json')
        assert value['complete'] and value['lost'] == 0 and value['clr_events'] > 0
        assert value['protocol'] == 'all-event-records-v1'
        assert value['input_sha256'] == pin(BASE / (name.split('-')[0] + '-capture/capture.nettrace'))['sha256']


supervisor.command_for = command_for
supervisor.after = after
if __name__ == '__main__': raise SystemExit(supervisor.main())
