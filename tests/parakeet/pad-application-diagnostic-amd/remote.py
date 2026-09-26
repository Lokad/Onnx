"""Adapt the existing bounded supervisor with startup attachment and release."""
import shutil
import remote_base as supervisor
from remote_base import BASE, DOTNET, FLAGS, read, pin, save, live, idle
from scope import compiled
from application_protocol import validate_records
from phase_audit import attribute


def command_for(name, spec):
    if name == 'sdk-version': return [DOTNET, '--version'], True, 2
    if name == 'tracer-version': return [DOTNET, BASE/'tracer/dotnet-trace.dll', '--version'], False, 0
    if name.startswith(('producer-', 'bridge-')):
        kind, action = name.split('-')
        project = 'consumer/SampledAudio.csproj' if kind == 'producer' else 'bridge/Bridge.csproj'
        command = [DOTNET, action, BASE/'source'/project, *FLAGS]
        if kind == 'producer': command += ['-p:FrozenProductDirectory='+str(BASE/'runtimes/reference')]
        command += ['--source', spec['feed'], '--packages', BASE/'packages'] if action == 'restore' else ['-c','Release','--no-restore','--disable-build-servers']
        return command, True, 2
    role, action = name.split('-'); assert role in ['current', 'candidate']
    if action == 'inventory':
        return [DOTNET, BASE/'source/bridge/bin/Release/net10.0/Bridge.dll', BASE/'runtimes/reference',
                BASE/'runtimes'/role, BASE/name/'instructions.json'], False, 2
    if action == 'capture':
        assert read(BASE/'candidate-inventory/review.json')['passed'] and read(BASE/'current-inventory/review.json')['passed']
        # The consumer itself creates its output directory.
        return [DOTNET, BASE/'runtimes'/role/'SampledAudio.dll', BASE/'assets', BASE/'manifest.json',
                BASE/name/'requests', 'timing', 'sampled'], False, 2
    assert action == 'export'
    return [DOTNET, BASE/'export-runtime/DispatchEventsExport.dll', BASE/(role+'-capture/capture.nettrace'),
            BASE/name/'events'], False, 0


def job_environment(name, spec, env):
    if not name.endswith('-capture'): return env
    role = name.split('-')[0]
    return dict(env, PARAKEET_PHASE_MODE='wall',
        PARAKEET_PHASE_CORE_SHA=spec['products'][role]['Lokad.Onnx.dll']['sha256'],
        PARAKEET_PHASE_DATA_SHA=spec['observer']['sha256'])


def after(name, spec, row):
    if name == 'sdk-version': assert (BASE/'logs/sdk-version.stdout').read_text().strip().endswith('10.0.204')
    if name == 'producer-build':
        folder = BASE/'source/consumer/bin/Release/net10.0'
        built = dict(passed=True, files={}, exporter=spec['exporter'], consumer=pin(folder/'SampledAudio.dll'))
        for role in ['current', 'candidate']:
            for suffix in ['dll','deps.json','runtimeconfig.json']:
                source = folder/('SampledAudio.'+suffix); target = BASE/'runtimes'/role/source.name
                assert not target.exists(); shutil.copy2(source, target)
                built['files'][target.relative_to(BASE).as_posix()] = pin(target)
        save(BASE/'built.json', built)
    if name.endswith('-inventory'):
        original, = [r for r in read(BASE/'evidence/observer-instructions.json')['observations'] if r['assembly'] == 'Lokad.Onnx.Data.dll']
        result = compiled(read(BASE/name/'instructions.json'), name.split('-')[0], spec, read(BASE/'built.json'), original)
        save(BASE/name/'review.json', dict(**result, inventory=pin(BASE/name/'instructions.json')))
    if name.endswith('-capture'):
        role = name.split('-')[0]; folder = BASE/name/'requests'
        value = read(folder/'result.json'); ready = read(folder/'ready.json')
        enabled = read(folder/'startup-enabled.json'); startup = read(folder/'startup-ready.json')
        assert ready['pid'] == enabled['pid'] == startup['pid'] == row['processes']['worker']['pid']
        assert ready['warmup_records'] == 20 and startup['counter'] < enabled['counter'] < value['records'][0]['start_ticks']
        validate_records(value, read(BASE/'manifest.json'), 'timing')
        assert value['passed'] and value['sampled'] and value['runtime'] == '.NET 10.0.8' and value['flags'] == {}
        assert value['core_sha256'] == spec['products'][role]['Lokad.Onnx.dll']['sha256']
        assert value['data_sha256'] == spec['observer']['sha256']
        assert value['runner_sha256'] == read(BASE/'built.json')['consumer']['sha256']
        assert value['manifest_sha256'] == pin(BASE/'manifest.json')['sha256']
        assert len(read(folder/'clock-anchors.json')['anchors']) == 161
        attribute(value, folder, 'wall')
        assert (BASE/name/'capture.nettrace').stat().st_size > 0
    if name.endswith('-export'):
        value = read(BASE/name/'events/summary.json')
        assert value['complete'] and value['lost'] == 0 and value['clr_events'] > 0
        assert value['protocol'] == 'all-event-records-v1'
        assert value['input_sha256'] == pin(BASE/(name.split('-')[0]+'-capture/capture.nettrace'))['sha256']


supervisor.command_for = command_for
supervisor.job_environment = job_environment
supervisor.after = after
if __name__ == '__main__': raise SystemExit(supervisor.main())
