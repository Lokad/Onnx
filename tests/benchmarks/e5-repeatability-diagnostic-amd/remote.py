"""Reuse the existing bounded event supervisor for eight fixed e5 processes."""
import gzip
import hashlib
import shutil
import remote_base as supervisor
from remote_base import BASE, DOTNET, FLAGS, read, pin, save, live, idle
from protocol import ROLES, KEYS
from checks import compiled_scope


def command_for(name, spec):
    if name=='sdk-version': return [DOTNET,'--version'],True,2
    if name=='exporter-roundtrip': return [DOTNET,BASE/'export-runtime/DispatchEventsExport.dll',BASE/'roundtrip/capture.nettrace',BASE/name/'events'],False,0
    if name=='tracer-version': return [DOTNET,BASE/'tracer/dotnet-trace.dll','--version'],False,0
    if name=='consumer-inventory': return [DOTNET,BASE/'bridge/Bridge.dll',BASE/'previous',BASE/'runtimes/current',BASE/name/'instructions.json'],False,2
    if name.startswith(('producer-','exporter-')):
        kind, action = name.split('-')
        project = BASE/'source'/('consumer/Producer.csproj' if kind=='producer' else 'exporter/Exporter.csproj')
        command = [DOTNET,action,project,*FLAGS]
        command += ['--source',spec['feed'],'--packages',BASE/'packages'] if action=='restore' else ['-c','Release','--no-restore','--disable-build-servers']
        return command,True,2
    role, action = name.split('-'); assert role in ROLES
    product = ROLES[role]
    if action=='capture':
        return [DOTNET,BASE/'runtimes'/product/'ReleaseBenchmark.dll',BASE/f'cases-{product}.json',KEYS[role],BASE/name/'output','timing'],False,2
    assert action=='export'
    return [DOTNET,BASE/'export-runtime/DispatchEventsExport.dll',BASE/(role+'-capture/capture.nettrace'),BASE/name/'events'],False,0


def after(name, spec, row):
    if name=='sdk-version': assert (BASE/'logs/sdk-version.stdout').read_text().strip().endswith('10.0.204')
    if name in ['producer-build','exporter-build']:
        built = read(BASE/'built.json') if (BASE/'built.json').exists() else dict(passed=True,files={})
        if name=='producer-build':
            folder = BASE/'source/consumer/bin/Release/net10.0'
            assert pin(folder/'Lokad.Onnx.dll') == spec['products']['current']['Lokad.Onnx.dll']
            for product in ['current','candidate']:
                for suffix in ['dll','deps.json','runtimeconfig.json']:
                    source = folder/('ReleaseBenchmark.'+suffix); target = BASE/'runtimes'/product/source.name
                    assert not target.exists(); shutil.copy2(source,target); built['files'][target.relative_to(BASE).as_posix()] = pin(target)
            built['consumer'] = pin(folder/'ReleaseBenchmark.dll')
        else:
            folder = BASE/'source/exporter/bin/Release/net10.0'; target = BASE/'export-runtime'; shutil.copytree(folder,target)
            for p in target.rglob('*'):
                if p.is_file(): built['files'][p.relative_to(BASE).as_posix()] = pin(p)
            built['exporter'] = pin(target/'DispatchEventsExport.dll')
        save(BASE/'built.json',built)
    if name=='consumer-inventory':
        save(BASE/name/'review.json',compiled_scope(read(BASE/name/'instructions.json'),spec,read(BASE/'built.json')))
    if name=='exporter-roundtrip':
        summary=read(BASE/name/'events/summary.json')
        assert summary['complete'] and summary['lost']==0 and summary['protocol']=='all-event-records-v1'
        assert summary['input_sha256']==pin(BASE/'roundtrip/capture.nettrace')['sha256']
        digest=hashlib.sha256();size=0
        with gzip.open(BASE/name/'events/events.jsonl.gz','rb') as stream:
            while chunk:=stream.read(1024**2):digest.update(chunk);size+=len(chunk)
        actual=dict(bytes=size,sha256=digest.hexdigest());assert actual==spec['roundtrip']['raw']
        assert summary['recorded']==spec['roundtrip']['events']
        save(BASE/name/'review.json',dict(passed=True,exact_original_events=actual,events=summary['recorded']))
    if name.endswith('-capture'):
        role = name.split('-')[0]; product = ROLES[role]
        value = read(BASE/name/'output/result.json'); diagnostic = read(BASE/name/'output/diagnostic.json')
        ready = read(BASE/name/'ready.json'); enabled = read(BASE/name/'collector-enabled.json')
        assert value['passed'] and value['mode']=='timing' and diagnostic['diagnosticOnly']
        assert value['pid']==diagnostic['pid']==row['processes']['worker']['pid']==ready['pid']==enabled['pid']
        assert diagnostic['nativeThread']==ready['native_thread'] and ready['counter']<enabled['counter']
        assert value['runtime']=='10.0.8' and value['flags']=={} and value['key']==KEYS[role]
        assert value['consumer']==read(BASE/'built.json')['consumer']['sha256']
        assert value['core']==spec['products'][product]['Lokad.Onnx.dll']['sha256']
        assert value['calls']==len(value['clocks'])==len(diagnostic['clocks'])==780
        for i, (clock, observation) in enumerate(zip(value['clocks'],diagnostic['clocks'],strict=True)):
            assert clock['index']==observation['index']==i and clock['warmup']==(i<600)
            assert clock['ticks']==observation['end']-observation['start']>0
        expected = read(BASE/f'evidence/original-{KEYS[role]}-{product}.json')
        assert value['arrays']==expected['arrays'] and value['inputs_unchanged'] and value['held_outputs_unchanged']
        assert (BASE/name/'capture.nettrace').stat().st_size>0
    if name.endswith('-export'):
        value = read(BASE/name/'events/summary.json')
        assert value['complete'] and value['lost']==0 and value['clr_events']>0 and value['protocol']=='all-event-records-v1'
        assert value['input_sha256']==pin(BASE/(name.split('-')[0]+'-capture/capture.nettrace'))['sha256']


supervisor.command_for = command_for
supervisor.after = after
if __name__=='__main__': raise SystemExit(supervisor.main())
