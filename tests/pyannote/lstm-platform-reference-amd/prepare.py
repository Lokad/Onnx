"""Freeze qualified binaries and exact AMD SDK/native dependencies before workers."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import traceback
from protocol import JOBS, LIMITS, pin, read, save

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/pyannote-lstm-platform-reference-amd-20260922'
LOCAL=ROOT/'artifacts/pyannote-lstm-input-blocks-v6-20260922'
PRODUCT=ROOT/'artifacts/pyannote-lstm-input-blocks-v2-20260922'
FIXTURES=ROOT/'artifacts/pyannote-lstm-input-fixtures-v3-20260922'
PREVIOUS=ROOT/'artifacts/pyannote-lstm-input-blocks-amd-20260922'
MONITOR=ROOT/'tests/parakeet/packing-budgets/common.py'
spec=importlib.util.spec_from_file_location('lstm_amd_monitor',MONITOR);monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)


def previous_closed():
    for folder,digest in [(LOCAL,'d00b9343dbb5b33eb2bb9db3fa6f139ca12afab910afa370c4421c6bf4a7d683'),
                          (FIXTURES,'c64da99fa9e6cbae1af930b6560096c4d9fdd3e7efdf6c07fb6c4e0e53184f7b')]:
        assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
        for identity in proof['identities']:monitor.terminal(identity)
    monitor.verify(read(LOCAL/'inputs.json')['files']);monitor.verify(read(LOCAL/'binaries.json')['files'])
    assert pin(PREVIOUS/'failure-closed.json')['sha256']=='32a258cb1da52f0499e022dd3d0aa787db496fd1affe5c43c68a22093ffb9b13'
    receipt=read(PREVIOUS/'collected/collection.json');assert receipt['terminal'] and receipt['code']==1 and receipt['input_error'] is None
    owner=receipt['identities'][0];assert owner==dict(pid=720767,birth=1790097478.29)
    for name,wanted in read(PREVIOUS/'failure-closed.json')['files'].items():assert pin(PREVIOUS/name)==wanted,name
    return owner


def prepare():
    assert not BASE.exists();owner=previous_closed();prior=read(PREVIOUS/'payload/payload.json');env={k:prior[k] for k in ['external','interpreter','packages']}
    BASE.mkdir();payload=BASE/'payload';payload.mkdir();(payload/'tools').mkdir();(payload/'fixtures').mkdir()
    for role,folder in [('selected','selected-runtime')]:shutil.copytree(LOCAL/folder,payload/'runtime'/role)
    for folder in ['output','native']:shutil.copytree(FIXTURES/folder,payload/'fixtures'/folder)
    for name in ['protocol.py','checks.py','remote.py','native.py']:shutil.copy2(TOOLS/name,payload/'tools'/name)
    shutil.copy2(ROOT/'PLAN.md',payload/'prospective-plan.md')
    cores={role:pin(payload/'runtime'/role/'Lokad.Onnx.dll') for role in ['selected']}
    build=BASE/'consumer';build.mkdir();(BASE/'logs').mkdir();(BASE/'output').mkdir()
    shutil.copy2(TOOLS/'ModelReplay.cs',build/'ModelReplay.cs')
    project=build/'ModelReplay.csproj'
    project.write_text('<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><AssemblyName>LstmModelReplay</AssemblyName><ImplicitUsings>enable</ImplicitUsings><Nullable>enable</Nullable></PropertyGroup><ItemGroup><Reference Include="Lokad.Onnx"><HintPath>../payload/runtime/selected/Lokad.Onnx.dll</HintPath></Reference><Reference Include="Google.Protobuf"><HintPath>../payload/runtime/selected/Google.Protobuf.dll</HintPath></Reference></ItemGroup></Project>',encoding='utf8')
    # Copy the selected product unchanged; replace only the standalone diagnostic consumer.
    for suffix in ['dll','deps.json','runtimeconfig.json']:(payload/'runtime/selected'/('LstmModelReplay.'+suffix)).unlink()
    inputs={p.as_posix():pin(p) for folder in [payload,build,TOOLS] for p in folder.rglob('*') if p.is_file()}
    for p in [LOCAL/'closed.json',FIXTURES/'closed.json',PREVIOUS/'failure-closed.json',MONITOR,ROOT/'tests/parakeet/portable-models/common.py']:inputs[p.as_posix()]=pin(p)
    save(BASE/'inputs.json',dict(files=inputs,no_product_rebuild=True))
    monitor.BASE=BASE;own=monitor.psutil.Process();state=dict(complete=False,code=None,supervisor=dict(pid=own.pid,birth=own.create_time()),runs=[])
    path=BASE/'controller.json';save(path,state)
    flags=monitor.FLAGS+['-p:NuGetAudit=false'];feed=ROOT/'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
    try:
        for name,cmd in [('restore',['dotnet','restore',project,*flags,'--source',feed,'--packages',BASE/'packages']),('build',['dotnet','build',project,'-c','Release',*flags,'--no-restore','--disable-build-servers'])]:
            monitor.worker(state,path,name,cmd,ROOT,[0],8,8,900,True,build)
        for suffix in ['dll','deps.json','runtimeconfig.json']:shutil.copy2(build/'bin/Release/net10.0'/('LstmModelReplay.'+suffix),payload/'runtime/selected'/('LstmModelReplay.'+suffix))
        consumer=pin(payload/'runtime/selected/LstmModelReplay.dll')
        monitor.worker(state,path,'local',['dotnet',payload/'runtime/selected/LstmModelReplay.dll',payload/'fixtures',BASE/'output/result.json',cores['selected']['sha256'],'selected','256'],ROOT,[0],12,8,900,False,BASE/'output')
        from checks import check_result
        result=read(BASE/'output/result.json');assert result['pid']==state['runs'][-1]['worker']['pid']
        local=check_result(result,'selected','256',dict(cores=cores,consumer=consumer),payload,BASE/'output',local=True)
        assert local['windows_changed']==0 and local['windows_maximum']==0
        save(BASE/'local-review.json',local);monitor.verify(inputs);state['code']=0
    except BaseException:state.update(code=1,error=traceback.format_exc());raise
    finally:state['complete']=True;save(path,state)

    manifest=dict(passed=True,limits=LIMITS,jobs=JOBS,previous_owner=owner,boot_time=1789634288.0,**env,cores=cores,consumer=consumer,
        files={p.relative_to(payload).as_posix():pin(p) for p in payload.rglob('*') if p.is_file()},
        scope='Selected-product platform reference diagnostic: all captured calls at AVX2/AVX512/scalar and fresh AMD ORT1.29. Quantify every Windows difference. No candidate admission, timing or codegen claim.')
    save(payload/'payload.json',manifest)
    files={p.relative_to(ROOT).as_posix():pin(p) for p in [*TOOLS.iterdir(),MONITOR,LOCAL/'closed.json',PRODUCT/'failure-closed.json',FIXTURES/'closed.json',PREVIOUS/'failure-closed.json',PREVIOUS/'collected/collection.json'] if p.is_file()}
    for p in TOOLS.glob('*.py'):ast.parse(p.read_text(),str(p))
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as tar:
        for p in sorted(payload.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(payload).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=files,payload=pin(payload/'payload.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(payload=pin(payload/'payload.json'),archive=pin(BASE/'payload.tar.gz'),cores=cores,jobs=JOBS,external=len(env['external']))))


if __name__=='__main__':prepare()
