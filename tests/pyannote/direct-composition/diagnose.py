"""Inspect unmodified v2 arithmetic at the first real-caller mismatch."""
import shutil
import traceback
import successor

c=successor.configured()
ROOT,TOOLS=c['ROOT'],c['TOOLS'];prior=c['BASE'];monitor=c['monitor']
pin,read,save,verify,terminal,psutil=[c[k] for k in ['pin','read','save','verify','terminal','psutil']]
BASE=ROOT / 'artifacts/pyannote-direct-caller-diagnostic-20260922';monitor.BASE=BASE
assert not BASE.exists()
assert pin(prior / 'failure-closed.json')['sha256']=='0e74d376d59407564852c1b06345d280d66a9323854729cdfbdb90f1ad934e6f'
proof=read(prior / 'failure-closed.json');verify(proof['files'])
for i in proof['identities']:terminal(i)
BASE.mkdir();(BASE / 'logs').mkdir();source=BASE / 'source';source.mkdir()
text=(TOOLS / 'CallerProbe.cs').read_text(encoding='utf8')
old='''                    for (int i = 0; i < wanted.Length; i++)
                        Require'''
new='''                    for (int i = 0; i < wanted.Length; i++)
                    {
                        if (BitConverter.SingleToInt32Bits(wanted[i]) != BitConverter.SingleToInt32Bits(actual[i]))
                            Diagnose(args[2] + ".mismatch.json", baseline, input, weights, bias,
                                rows, reduction, block, groups, batch, i, wanted[i], actual[i]);
                        Require'''
assert text.count(old)==1;text=text.replace(old,new)
old='''                    if (batch == 1) Require'''
assert text.count(old)==1;text=text.replace(old,'''                    }
                    if (batch == 1) Require''')
needle='    static int Main(string[] args)'
assert text.count(needle)==1;text=text.replace(needle,(TOOLS / 'CallerDiagnostic.txt').read_text(encoding='utf8')+needle)
(source / 'Program.cs').write_text(text,encoding='utf8')
project=source / 'Diagnostic.csproj'
project.write_text(f'''<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><Nullable>enable</Nullable><AllowUnsafeBlocks>true</AllowUnsafeBlocks></PropertyGroup><ItemGroup><Reference Include="Lokad.Onnx"><HintPath>{prior / 'runtime/Lokad.Onnx.dll'}</HintPath></Reference></ItemGroup></Project>''',encoding='utf8')
files={c['rel'](p):pin(p) for p in [prior / 'failure-closed.json',*source.iterdir(),TOOLS / 'diagnose.py',TOOLS / 'CallerDiagnostic.txt',TOOLS / 'CallerProbe.cs']}
save(BASE / 'inputs.json',dict(files=files))
own=psutil.Process();state=dict(complete=False,code=None,supervisor=dict(pid=own.pid,birth=own.create_time()),runs=[])
path=BASE / 'processes.json';save(path,state);flags=monitor.FLAGS+['-p:NuGetAudit=false']
try:
    jobs=[('restore',['dotnet','restore',project,*flags,'--source',c['FEED'],'--packages',BASE / 'packages'],[0]),
        ('build',['dotnet','build',project,'-c','Release',*flags,'--no-restore','--disable-build-servers'],[0]),
        ('diagnostic',['dotnet',source / 'bin/Release/net10.0/Diagnostic.dll',c['CONTROL'],c['COMPONENT'] / 'payload/shapes.json',BASE / 'result.json',
            'normal',pin(prior / 'runtime/Lokad.Onnx.dll')['sha256']],[3762504530])]
    for name,command,expected in jobs:
        monitor.worker(state,path,name,command,ROOT,expected,8,8,900,True,BASE)
        print(name,'terminal as expected',flush=True)
    assert pin(source / 'bin/Release/net10.0/Lokad.Onnx.dll')==pin(prior / 'runtime/Lokad.Onnx.dll')
    result=read(BASE / 'result.json.mismatch.json');assert result['diagnostic_only']
    verify(files);save(BASE / 'diagnosed.json',dict(passed=True,expected_mismatch=True,files=files,details=pin(BASE / 'result.json.mismatch.json')))
    state['code']=0
except BaseException:
    state.update(code=1,error=traceback.format_exc());raise
finally:
    state['complete']=True;save(path,state)
