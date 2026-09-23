"""Adapt the retained bounded event supervisor to two identical GPT-2 products."""
import remote_base as supervisor
from remote_base import BASE,DOTNET,FLAGS,read,pin,save,live,idle
import shutil

def command_for(name,spec):
    if name=='sdk-version':return [DOTNET,'--version'],True,2
    if name=='tracer-version':return [DOTNET,BASE/'tracer/dotnet-trace.dll','--version'],False,0
    if name.startswith(('producer-','exporter-')):
        kind,action=name.split('-')
        project=BASE/'source'/('consumer/Producer.csproj' if kind=='producer' else 'exporter/Exporter.csproj')
        command=[DOTNET,action,project,*FLAGS]
        command+=['--source',spec['feed'],'--packages',BASE/'packages'] if action=='restore' else ['-c','Release','--no-restore','--disable-build-servers']
        return command,True,2
    role,action=name.split('-');assert role in ['a','b']
    if action=='capture':return [DOTNET,BASE/'runtimes'/role/'GraphStartupDiagnostic.dll',BASE/'cases.json','gpt2',BASE/name/'output','diagnostic'],False,2
    assert action=='export'
    return [DOTNET,BASE/'export-runtime/DispatchEventsExport.dll',BASE/(role+'-capture/capture.nettrace'),BASE/name/'events'],False,0

def after(name,spec,row):
    if name=='sdk-version':assert (BASE/'logs/sdk-version.stdout').read_text().strip().endswith('10.0.204')
    if name in ['producer-build','exporter-build']:
        built=read(BASE/'built.json') if (BASE/'built.json').exists() else dict(passed=True,files={})
        if name=='producer-build':
            folder=BASE/'source/consumer/bin/Release/net10.0'
            for role in ['a','b']:
                for suffix in ['dll','deps.json','runtimeconfig.json']:
                    source=folder/('GraphStartupDiagnostic.'+suffix);target=BASE/'runtimes'/role/source.name
                    assert not target.exists();shutil.copy2(source,target);built['files'][target.relative_to(BASE).as_posix()]=pin(target)
            built['consumer']=pin(folder/'GraphStartupDiagnostic.dll')
        else:
            folder=BASE/'source/exporter/bin/Release/net10.0';target=BASE/'export-runtime';shutil.copytree(folder,target)
            for p in target.rglob('*'):
                if p.is_file():built['files'][p.relative_to(BASE).as_posix()]=pin(p)
            built['exporter']=pin(target/'DispatchEventsExport.dll')
        save(BASE/'built.json',built)
    if name.endswith('-capture'):
        value=read(BASE/name/'output/result.json');ready=read(BASE/name/'ready.json');enabled=read(BASE/name/'collector-enabled.json')
        assert value['passed'] and value['diagnosticOnly'] and value['mode']=='diagnostic'
        assert value['pid']==row['processes']['worker']['pid']==ready['pid']==enabled['pid']
        assert value['nativeThread']==ready['native_thread'] and ready['counter']<enabled['counter']
        assert value['runtime']=='10.0.8' and value['flags']=={} and value['key']=='gpt2'
        assert value['consumer']==read(BASE/'built.json')['consumer']['sha256']
        assert value['core']==spec['products']['current']['Lokad.Onnx.dll']['sha256']
        assert value['calls']==len(value['clocks'])==1200
        expected=read(BASE/'evidence/original-gpt2.json')
        assert value['arrays']==expected['arrays'] and value['inputs_unchanged'] and value['held_outputs_unchanged']
        assert (BASE/name/'capture.nettrace').stat().st_size>0
    if name.endswith('-export'):
        value=read(BASE/name/'events/summary.json');assert value['complete'] and value['lost']==0 and value['clr_events']>0
        assert value['protocol']=='all-event-records-v1'
        assert value['input_sha256']==pin(BASE/(name.split('-')[0]+'-capture/capture.nettrace'))['sha256']

supervisor.command_for=command_for
supervisor.after=after
if __name__=='__main__':raise SystemExit(supervisor.main())
