"""Run six bounded jobs using unchanged compiled diagnostic instruments."""
import remote_base as supervisor
from remote_base import BASE, DOTNET, read, pin, save, live, idle
from checks import rows


def command_for(name, spec):
    if name == 'sdk-version': return [DOTNET, '--version'], True, 2
    if name == 'tracer-version': return [DOTNET, BASE / 'tracer/dotnet-trace.dll', '--version'], False, 0
    role, action = name.split('-'); assert role in ['current', 'candidate']
    if action == 'capture':
        return [DOTNET, BASE / 'runtimes' / role / 'ParakeetDispatchEvents.dll', BASE, role,
                ['current', 'candidate'].index(role), BASE / name / 'result.json'], False, 2
    assert action == 'export'
    return [DOTNET, BASE / 'export-runtime/DispatchEventsExport.dll', BASE / (role + '-capture/capture.nettrace'), BASE / name / 'events'], False, 0


def after(name, spec, row):
    if name == 'sdk-version': assert (BASE / 'logs/sdk-version.stdout').read_text().strip().endswith('10.0.204')
    if name.endswith('-capture'):
        role = name.split('-')[0]; value = read(BASE / name / 'result.json')
        rows(value, read(BASE / 'evidence/capture.json'))
        assert value['pid'] == row['processes']['worker']['pid'] and value['runtime'] == '10.0.8'
        assert value['role'] == role and value['sequence'] == ['current', 'candidate'].index(role)
        assert value['core_sha256'] == spec['products'][role]['Lokad.Onnx.dll']['sha256']
        ready = read(BASE / name / 'ready.json'); enabled = read(BASE / name / 'collector-enabled.json')
        assert value['pid'] == ready['pid'] == enabled['pid'] and value['nativeThread'] == ready['native_thread']
        assert ready['counter'] < enabled['counter'] < value['rows'][0]['clocks'][0]['marker']
        assert value['protocol'] == 'parakeet-dispatch-events-v1' and value['flags'] == {}
        assert value['assembly'] == spec['capture_consumer']['sha256']
        assert (BASE / name / 'capture.nettrace').stat().st_size > 0
    if name.endswith('-export'):
        value = read(BASE / name / 'events/summary.json')
        assert value['complete'] and value['lost'] == 0 and value['clr_events'] > 0
        assert value['protocol'] == 'all-event-records-v1'
        assert value['input_sha256'] == pin(BASE / (name.split('-')[0] + '-capture/capture.nettrace'))['sha256']


supervisor.command_for = command_for
supervisor.after = after
if __name__ == '__main__': raise SystemExit(supervisor.main())
