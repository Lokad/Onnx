"""Check whether the accepted caller itself preserves NaN payloads while warming."""
import shutil
import traceback
import successor

c=successor.configured();ROOT,TOOLS=c['ROOT'],c['TOOLS'];monitor=c['monitor']
pin,read,save,verify,terminal,psutil=[c[k] for k in ['pin','read','save','verify','terminal','psutil']]
BASE=ROOT / 'artifacts/pyannote-baseline-nan-20260922';monitor.BASE=BASE
assert not BASE.exists();BASE.mkdir();(BASE / 'logs').mkdir();source=BASE / 'source';source.mkdir()
shutil.copy2(TOOLS / 'BaselineNaN.cs',source / 'Program.cs')
project=source / 'BaselineNaN.csproj'
project.write_text(f'''<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><Nullable>enable</Nullable></PropertyGroup><ItemGroup><Reference Include="Lokad.Onnx"><HintPath>{c['CONTROL'] / 'Lokad.Onnx.dll'}</HintPath></Reference></ItemGroup></Project>''',encoding='utf8')
files={c['rel'](p):pin(p) for p in [*source.iterdir(),TOOLS / 'baseline_nan.py',TOOLS / 'BaselineNaN.cs',c['CONTROL'] / 'Lokad.Onnx.dll']}
save(BASE / 'inputs.json',dict(files=files))
own=psutil.Process();state=dict(complete=False,code=None,supervisor=dict(pid=own.pid,birth=own.create_time()),runs=[])
path=BASE / 'processes.json';save(path,state);flags=monitor.FLAGS+['-p:NuGetAudit=false']
try:
    jobs=[('restore',['dotnet','restore',project,*flags,'--source',c['FEED'],'--packages',BASE / 'packages']),
        ('build',['dotnet','build',project,'-c','Release',*flags,'--no-restore','--disable-build-servers'])]
    jobs += [(f'block-{k}',['dotnet',source / 'bin/Release/net10.0/BaselineNaN.dll',str(k),BASE / f'block-{k}.json']) for k in [1,7]]
    for name,command in jobs:
        monitor.worker(state,path,name,command,ROOT,[0],8,8,900,True,BASE)
        print(name,'passed',flush=True)
    assert pin(source / 'bin/Release/net10.0/Lokad.Onnx.dll')==pin(c['CONTROL'] / 'Lokad.Onnx.dll')
    verify(files);state['code']=0
except BaseException:
    state.update(code=1,error=traceback.format_exc());raise
finally:
    state['complete']=True;save(path,state)
