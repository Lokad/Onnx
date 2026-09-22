"""Revalidate the unchanged v1 product against the demonstrated float contract."""
import shutil
import sys
import traceback
import prepare as original

ROOT,TOOLS=original.ROOT,original.TOOLS
BASE=ROOT / 'artifacts/pyannote-direct-composition-acceptance-20260922'
PRIOR=original.BASE


def configured():
    source=(TOOLS / 'prepare.py').read_text(encoding='utf8')
    assert source.count('pyannote-direct-composition-20260922')==1
    source=source.replace('pyannote-direct-composition-20260922','pyannote-direct-composition-acceptance-20260922')
    namespace=dict(__name__='direct_acceptance',__file__=str(TOOLS / 'prepare.py'))
    exec(compile(source,str(TOOLS / 'prepare.py'),'exec'),namespace)
    return namespace


def prepare():
    c=configured();pin,read,save,verify,terminal=[c[k] for k in ['pin','read','save','verify','terminal']]
    assert not BASE.exists()
    proof=read(PRIOR / 'failure-closed.json');assert proof['passed'] and proof['expected_failure'];verify(proof['files'])
    for i in proof['identities']:terminal(i)
    diagnostic=ROOT / 'artifacts/pyannote-baseline-nan-20260922/closed.json'
    assert pin(diagnostic)['sha256']=='abecc174db088ba57bd5bd973b37a17a50cb995dd07c2f7a9c4d941d6bc26fdb'
    control=read(diagnostic);assert control['passed'];verify(control['files'])
    for i in control['identities']:terminal(i)
    assert control['result']['block7']['differing_nan_payloads']==6
    BASE.mkdir();(BASE / 'logs').mkdir()
    # Copy the complete normal build; no product or test source is recompiled.
    for folder in ['source','runtime','bridge']:
        shutil.copytree(PRIOR / folder,BASE / folder)
    for name in ['instructions.json','instruction-review.json','candidate.patch']:
        shutil.copy2(PRIOR / name,BASE / name)
    assert c['review']()==read(BASE / 'instruction-review.json')
    caller=BASE / 'caller';caller.mkdir()
    text=(TOOLS / 'CallerProbe.cs').read_text(encoding='utf8')
    old='''                var wanted = (float[])initial.Clone(); var actual = (float[])initial.Clone();'''
    new=old+'\n                var payloadDifferences = new List<object>();'
    assert text.count(old)==1;text=text.replace(old,new)
    old='''                    for (int i = 0; i < wanted.Length; i++)
                        Require(BitConverter.SingleToInt32Bits(wanted[i]) == BitConverter.SingleToInt32Bits(actual[i]),'''
    new='''                    for (int i = 0; i < wanted.Length; i++)
                    {
                        bool same = BitConverter.SingleToInt32Bits(wanted[i]) == BitConverter.SingleToInt32Bits(actual[i]);
                        if (!same && float.IsNaN(wanted[i]) && float.IsNaN(actual[i]))
                        {
                            payloadDifferences.Add(new { batch, index = i,
                                baseline = BitConverter.SingleToInt32Bits(wanted[i]).ToString("x8"),
                                candidate = BitConverter.SingleToInt32Bits(actual[i]).ToString("x8") });
                            same = true;
                        }
                        Require(same,'''
    assert text.count(old)==1;text=text.replace(old,new)
    old='                    if (batch == 1) Require'
    assert text.count(old)==1;text=text.replace(old,'                    }\n'+old)
    old='nan_values = actual.Count(float.IsNaN), digest = Hash(actual)'
    new=old+', baseline_digest = Hash(wanted), nan_payload_differences = payloadDifferences'
    assert text.count(old)==1;text=text.replace(old,new)
    (caller / 'Program.cs').write_text(text,encoding='utf8')
    project=caller / 'Caller.csproj'
    project.write_text(f'''<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><Nullable>enable</Nullable></PropertyGroup><ItemGroup><Reference Include="Lokad.Onnx"><HintPath>{BASE / 'runtime/Lokad.Onnx.dll'}</HintPath></Reference></ItemGroup></Project>''',encoding='utf8')
    files={c['rel'](p):pin(p) for folder in [BASE / 'source',BASE / 'runtime',BASE / 'bridge',caller] for p in folder.rglob('*')
        if p.is_file() and 'obj' not in p.relative_to(folder).parts}
    for p in [PRIOR / 'failure-closed.json',diagnostic,*TOOLS.iterdir(),BASE / 'instructions.json',BASE / 'instruction-review.json',BASE / 'candidate.patch']:
        if p.is_file():files[c['rel'](p)]=pin(p)
    save(BASE / 'inputs.json',dict(passed=True,files=files,unchanged_initial_product=True,
        contract='Every non-NaN output bit exact, both engines NaN at each NaN position, every payload difference retained; all input/guard bits exact.'))
    own=c['psutil'].Process();state=dict(complete=False,code=None,supervisor=dict(pid=own.pid,birth=own.create_time()),runs=[])
    path=BASE / 'preparation.json';save(path,state);monitor=c['monitor'];flags=monitor.FLAGS+['-p:NuGetAudit=false']
    try:
        jobs=[('caller-restore',['dotnet','restore',project,*flags,'--source',c['FEED'],'--packages',BASE / 'packages']),
            ('caller-build',['dotnet','build',project,'-c','Release',*flags,'--no-restore','--disable-build-servers'])]
        for name,command in jobs:
            monitor.worker(state,path,name,command,BASE / 'source',[0],8,8,900,True,caller)
            print(name,'passed',flush=True)
        for mode in ['normal','disabled']:
            old_env=monitor.clean_env
            if mode=='disabled':monitor.clean_env=lambda:old_env() | {'DOTNET_EnableHWIntrinsic':'0'}
            try:
                monitor.worker(state,path,'caller-'+mode,['dotnet',caller / 'bin/Release/net10.0/Caller.dll',c['CONTROL'],
                    c['COMPONENT'] / 'payload/shapes.json',BASE / ('caller-'+mode+'.json'),mode,pin(BASE / 'runtime/Lokad.Onnx.dll')['sha256']],
                    BASE / 'source',[0],8,8,900,False,caller)
            finally:monitor.clean_env=old_env
            result=read(BASE / ('caller-'+mode+'.json'));assert result['passed'] and len(result['records'])==736
            print('caller-'+mode,'passed',flush=True)
        verify(files)
        files.update({c['rel'](p):pin(p) for p in (caller / 'bin').rglob('*') if p.is_file()})
        save(BASE / 'prepared.json',dict(passed=True,files=files,core=pin(BASE / 'runtime/Lokad.Onnx.dll'),data=pin(BASE / 'runtime/Lokad.Onnx.Data.dll'),
            caller_cases_per_mode=736,production_changed=False,models_qualified=False,performance_qualified=False,unchanged_initial_product=True))
        state['code']=0
    except BaseException:
        state.update(code=1,error=traceback.format_exc());raise
    finally:
        state['complete']=True;save(path,state)


if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ['prepare','qualify']
    if sys.argv[1]=='prepare':prepare()
    else:
        namespace=configured()
        source=(TOOLS / 'qualify.py').read_text(encoding='utf8').replace('from prepare import *','')
        exec(compile(source,str(TOOLS / 'qualify.py'),'exec'),namespace)
        namespace['main']()
