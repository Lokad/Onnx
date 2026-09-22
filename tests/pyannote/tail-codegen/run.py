"""Capture existing kernels without rebuilding or modifying their assemblies."""
from pathlib import Path
import shutil
import sys
import tarfile
import traceback

TOOLS = Path(__file__).resolve().parent
ORIGINAL = TOOLS.parent / 'direct-output'
sys.path.insert(0, str(ORIGINAL))
import common as c
c.BASE = c.ROOT / 'artifacts/pyannote-tail-codegen-20260922'
c.REMOTE = '/dev/shm/lokad-pyannote-tail-codegen-20260922'
c.monitor.BASE = c.BASE
FILTER = 'mm_unsafe_vectorized_intrinsics_3x4packed DirectOutput:*'
PRIORS = {
    'masked': ('pyannote-direct-output-v4-20260922', '7bfa418ab3ba005217c34dcf2ac80f859ca77c822e2fb662e779eff0912ce318'),
    'ordinary': ('pyannote-direct-output-store-v2-20260922', '66b2e4e8f92119b2fdd02412e50bf674e3b71444db3c2735e99c2e9293d3a524')}


def remote_source():
    source = (ORIGINAL / 'remote.py').read_text(encoding='utf8')
    old = """    command = [DOTNET,str(BASE / 'runtime/DirectOutputProbe.dll'),str(BASE / 'shapes.json'),
        name if name.startswith('validate') else name.split('-')[0],str(BASE / 'output' / (name+'.json'))]"""
    new = """    job = read(BASE / 'payload.json')['job_details'][name]
    command = [DOTNET,str(BASE / 'runtime/TailCodegen.dll'),str(BASE / job['probe']),str(BASE / job['shapes']),
        job['role'],job['probe_sha256'],str(BASE / 'output' / (name+'.json'))]"""
    changes = [(old,new),
        ("                if name=='validate-no-avx2': env['DOTNET_EnableAVX2']='0'", f"                env['DOTNET_JitDisasm']={FILTER!r}"),
        ("        if name=='validate-no-avx2': assert result['flags']==['DOTNET_EnableAVX2'] and not result['avx2'] and not result['avx512']\n        else: assert result['flags']==[] and result['avx2'] and result['avx512']",
         "        assert result['flags']==['DOTNET_JitDisasm'] and result['avx2'] and result['avx512']\n        assert result['probe']==job['probe_sha256'] and len(result['blocks'])==132\n        assert 'Tier1' in (BASE / 'logs' / (name+'.log')).read_text()")]
    for before,after in changes:
        assert source.count(before)==1,before
        source=source.replace(before,after)
    compile(source,'tail-codegen-remote','exec')
    return source


def prepare():
    assert not c.BASE.exists()
    for folder,digest in PRIORS.values():
        prior=c.ROOT / 'artifacts' / folder
        assert c.pin(prior / 'closed.json')['sha256']==digest
        c.verify(c.read(prior / 'closed.json')['files'])
    for folder in ['consumer','logs','output','payload/runtime','payload/tools','payload/probes']:
        (c.BASE / folder).mkdir(parents=True,exist_ok=True)
    payload=c.BASE / 'payload'; runtime=payload / 'runtime'
    jobs={}
    for role,(folder,_) in PRIORS.items():
        prior=c.ROOT / 'artifacts' / folder / 'payload'
        target=payload / 'probes' / role; target.mkdir()
        for name in ['DirectOutputProbe.dll']:
            shutil.copy2(prior / 'runtime' / name,target / name)
        shutil.copy2(prior / 'shapes.json',target / 'shapes.json')
        assert c.pin(prior / 'runtime/Lokad.Onnx.dll')['sha256']==c.CORE
        jobs[role]=dict(probe=f'probes/{role}/DirectOutputProbe.dll',shapes=f'probes/{role}/shapes.json',
            role='candidate',probe_sha256=c.pin(target / 'DirectOutputProbe.dll')['sha256'])
    jobs={'baseline':dict(jobs['masked'],role='baseline'),**jobs}
    shutil.copy2(c.ROOT / 'artifacts' / PRIORS['masked'][0] / 'payload/runtime/Lokad.Onnx.dll',runtime / 'Lokad.Onnx.dll')
    shutil.copy2(TOOLS / 'Driver.cs',c.BASE / 'consumer/Driver.cs')
    project=c.BASE / 'consumer/Driver.csproj'
    project.write_text(f'''<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><AssemblyName>TailCodegen</AssemblyName><Nullable>enable</Nullable><NuGetAudit>false</NuGetAudit></PropertyGroup><ItemGroup><Reference Include="Lokad.Onnx"><HintPath>{runtime / 'Lokad.Onnx.dll'}</HintPath></Reference></ItemGroup></Project>''',encoding='utf8')
    state=c.new_state(); path=c.BASE / 'preparation.json'; c.save(path,state)
    try:
        flags=c.monitor.FLAGS+['-p:NuGetAudit=false']
        for name,command in [('restore',['dotnet','restore',project,*flags,'--source',c.FEED,'--packages',c.BASE / 'packages']),
            ('build',['dotnet','build',project,'-c','Release',*flags,'--no-restore','--disable-build-servers'])]:
            c.monitor.worker(state,path,name,command,c.ROOT,[0],8,4,900,True,None)
        binary=project.parent / 'bin/Release/net10.0'
        assert c.pin(binary / 'Lokad.Onnx.dll')['sha256']==c.CORE
        for suffix in ['dll','deps.json','runtimeconfig.json']:
            shutil.copy2(binary / ('TailCodegen.'+suffix),runtime / ('TailCodegen.'+suffix))
        job=jobs['ordinary']; old_env=c.monitor.clean_env
        c.monitor.clean_env=lambda:old_env() | {'DOTNET_JitDisasm':FILTER}
        try:
            c.monitor.worker(state,path,'local-ordinary',['dotnet',runtime / 'TailCodegen.dll',payload / job['probe'],
                payload / job['shapes'],job['role'],job['probe_sha256'],c.BASE / 'output/local-ordinary.json'],
                c.ROOT,[0],8,2,900,False,c.BASE / 'output')
        finally: c.monitor.clean_env=old_env
        result=c.read(c.BASE / 'output/local-ordinary.json')
        assert result['passed'] and len(result['validation'])==3266 and len(result['blocks'])==132
        log=(c.BASE / 'logs/local-ordinary.log').read_text()
        assert 'Assembly listing for method DirectOutput:Multiply' in log and 'Tier1' in log
        (payload / 'tools/remote.py').write_text(remote_source(),encoding='utf8')
        external=c.read(c.ROOT / 'artifacts/pyannote-amd-profile-runtime-20260922.json')
        spec=dict(files={p.relative_to(payload).as_posix():c.pin(p) for p in payload.rglob('*') if p.is_file()},
            external=external['files'],boot_time=external['boot_time'],core=c.CORE,jobs=list(jobs),job_details=jobs,
            filter=FILTER,limits=dict(seconds=900,preflight_available=8*1024**3,preflight_tmpfs=3*1024**3,
                rss=2*1024**3,available=1024**3,tmpfs=1024**3,artifacts=1024**3))
        c.save(payload / 'payload.json',spec)
        with tarfile.open(c.BASE / 'payload.tar.gz','w:gz') as tar:
            for p in sorted(payload.rglob('*')):
                if p.is_file():tar.add(p,arcname=p.relative_to(payload).as_posix(),recursive=False)
        pins={c.rel(p):c.pin(p) for folder in [TOOLS,payload,c.BASE / 'consumer'] for p in folder.rglob('*')
            if p.is_file() and not {'obj','bin'}.intersection(p.relative_to(folder).parts)}
        for folder,_ in PRIORS.values():
            prior=c.ROOT / 'artifacts' / folder
            pins[c.rel(prior / 'closed.json')]=c.pin(prior / 'closed.json')
            pins.update(c.read(prior / 'closed.json')['files'])
        for p in [c.MONITOR,ORIGINAL / 'common.py',ORIGINAL / 'transport.py',ORIGINAL / 'remote.py',
            c.BASE / 'output/local-ordinary.json',c.BASE / 'logs/local-ordinary.log']:
            pins[c.rel(p)]=c.pin(p)
        c.save(c.BASE / 'prepared.json',dict(passed=True,files=pins,archive=c.pin(c.BASE / 'payload.tar.gz'),payload=c.pin(payload / 'payload.json')))
        state['code']=0; print('Prepared diagnostic with unchanged binaries',flush=True)
    except BaseException:
        state.update(code=1,error=traceback.format_exc()); raise
    finally:
        state['complete']=True;c.save(path,state)


if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ['prepare','stage','launch','observe','collect']
    if sys.argv[1]=='prepare':prepare()
    else:
        import transport
        getattr(transport,sys.argv[1])()
