"""Build a fixed complete-graph-call timer; preserve both product and layer-consumer assemblies."""
import ast
import importlib.util
import json
import io
import unittest
from pathlib import Path
import shutil
import tarfile
import traceback
from protocol import JOBS, LIMITS, pin, read, save
from score import iteration_manifest
from checks import check_result

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/pyannote-spatial-weight-screen-20260922'
CODEGEN = ROOT/'artifacts/pyannote-spatial-weight-codegen-20260922'
ROLES = ['production', 'candidate']
LOCAL = ROOT/'artifacts/pyannote-spatial-weight-reuse-20260922'
AMD = ROOT/'artifacts/pyannote-spatial-weight-reuse-amd-20260922'
PREVIOUS = ROOT/'artifacts/pyannote-blocked-spatial-product-amd-v3-20260922'
FIXTURES = ROOT/'artifacts/pyannote-blocked-spatial-fixtures-20260922/output'
FEED = ROOT/'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
MONITOR = ROOT/'tests/parakeet/packing-budgets/common.py'
spec = importlib.util.spec_from_file_location('spatial_weight_codegen_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec); spec.loader.exec_module(monitor); monitor.BASE = BASE


def previous_closed():
    for folder, digest in [(CODEGEN,'7c32ceccb3349efcde4c614ba00f10a1ea3d7c2ece57633d43609d6048cbd4bf'), (LOCAL,'b6b1ce3bfe8fe85f390ed4bc4af03f0090c9a3f1226d7c2d2edfde684900adf5'),
                           (AMD,'f7df7b7652a2dab086d2e364e2a1b52b472c471a0f1956bb40e5af4fa982b5aa'),
                           (PREVIOUS,'88ec9b71d6e7807440ccd65d6148ead8c585a0633ce755898e174b8501b77197')]:
        assert pin(folder/'closed.json')['sha256'] == digest
        proof = read(folder/'closed.json'); assert proof['passed']
        for name, wanted in proof['files'].items(): assert pin(folder/name) == wanted,name
    for identity in read(LOCAL/'closed.json')['identities']: monitor.terminal(identity)
    receipt = read(CODEGEN/'collected/collection.json'); assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
    if (BASE/'controller.json').exists():
        local = read(BASE/'controller.json'); assert local['complete'] and local['code'] == 0
        monitor.terminal(local['supervisor'])
        for row in local['runs']:
            for pid,birth in row['members'].items(): monitor.terminal(dict(pid=int(pid),birth=birth))
    return {k:receipt['identities'][0][k] for k in ['pid','birth']}


def prepare():
    assert not BASE.exists(); owner = previous_closed()
    BASE.mkdir()
    suite=unittest.defaultTestLoader.discover(str(TOOLS),pattern='test_score.py'); output=io.StringIO()
    tests=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
    save(BASE/'selftest.json',dict(passed=tests.wasSuccessful(),tests=tests.testsRun,output=output.getvalue()))
    assert tests.wasSuccessful() and tests.testsRun==12
    (BASE/'logs').mkdir(); (BASE/'output').mkdir(); consumer=BASE/'consumer'; consumer.mkdir()
    payload=BASE/'payload'; payload.mkdir(); (payload/'tools').mkdir(); (payload/'fixtures').mkdir()
    details={}
    for role, built in [('production',PREVIOUS/'payload/runtime'), ('candidate',LOCAL/'consumers/layers/bin/Release/net10.0')]:
        target=payload/'runtime'/role; shutil.copytree(built,target)
        details[role]=dict(core=pin(target/'Lokad.Onnx.dll'),probe=pin(target/'LayerGraphs.dll'))
    assert details['production']['core']['sha256']=='3c2f16b08856426d3dfeff07f1638dd76cee7f06b65bbee230e8e0789679206f'
    assert details['candidate']['core']['sha256']=='3ca0a2a5ff1b129272da05b679d539d865a3c6c20ee8625c24955e2a5f497c6a'
    shutil.copy2(AMD/'collected/layers-512/result.json',payload/'reference.json')
    shutil.copy2(FIXTURES/'result.json',payload/'fixtures/result.json')
    save(payload/'iterations.json',iteration_manifest(read(FIXTURES/'result.json')['calls']))
    shutil.copy2(TOOLS/'Driver.cs',consumer/'Driver.cs')
    project=consumer/'Screen.csproj'
    project.write_text('''<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><AssemblyName>SpatialWeightScreen</AssemblyName><Nullable>enable</Nullable></PropertyGroup><ItemGroup><Reference Include="Lokad.Onnx"><HintPath>../payload/runtime/production/Lokad.Onnx.dll</HintPath></Reference><Reference Include="Google.Protobuf"><HintPath>../payload/runtime/production/Google.Protobuf.dll</HintPath></Reference></ItemGroup></Project>''',encoding='utf8')
    for name in ['protocol.py','remote.py','checks.py','score.py']: shutil.copy2(TOOLS/name,payload/'tools'/name)
    shutil.copy2(ROOT/'.agent/m21-pyannote-spatial-weight-reuse-20260922.md',payload/'prospective-plan.md')
    inputs={p.as_posix():pin(p) for folder in [TOOLS,consumer,payload] for p in folder.rglob('*') if p.is_file()}
    for p in [MONITOR,ROOT/'tests/parakeet/portable-models/common.py',LOCAL/'closed.json',AMD/'closed.json',PREVIOUS/'closed.json',CODEGEN/'closed.json']:inputs[p.as_posix()]=pin(p)
    save(BASE/'inputs.json',dict(files=inputs,no_performance_measurement=True))
    own=monitor.psutil.Process();state=dict(complete=False,code=None,supervisor=dict(pid=own.pid,birth=own.create_time()),runs=[]);state_path=BASE/'controller.json';save(state_path,state)
    flags=monitor.FLAGS+['-p:NuGetAudit=false']
    try:
        for name,command in [('restore',['dotnet','restore',project,*flags,'--source',FEED,'--packages',BASE/'packages']),
                             ('build',['dotnet','build',project,'-c','Release',*flags,'--no-restore','--disable-build-servers'])]:
            monitor.worker(state,state_path,name,command,ROOT,[0],8,8,900,True,consumer);print(name,'passed',flush=True)
        built=consumer/'bin/Release/net10.0'
        for role in ROLES:
            for suffix in ['dll','deps.json','runtimeconfig.json']:shutil.copy2(built/('SpatialWeightScreen.'+suffix),payload/'runtime'/role/('SpatialWeightScreen.'+suffix))
        diagnostic=dict(job_details=details,driver=pin(built/'SpatialWeightScreen.dll'))
        for role in ROLES:
            runtime=payload/'runtime'/role; job=details[role]; destination=BASE/'output'/('local-'+role); destination.mkdir()
            monitor.worker(state,state_path,'local-'+role,['dotnet',runtime/'SpatialWeightScreen.dll','validate',role,FIXTURES,
                payload/'reference.json',payload/'iterations.json',destination,job['core']['sha256'],job['probe']['sha256'],'256'],ROOT,[0],12,8,900,False,BASE/'output')
            result=read(destination/'result.json');check_result(result,role,diagnostic,payload,8,False)
            assert result['pid']==state['runs'][-1]['worker']['pid'];print('local-'+role,'passed',flush=True)
        monitor.verify(inputs)
        old=read(AMD/'payload/payload.json')
        manifest=dict(passed=True,limits=LIMITS,jobs=JOBS,previous_owner=owner,boot_time=1789634288.0,
            interpreter=old['interpreter'],external=old['external'],fixture_directory=old['fixture_directory'],**diagnostic,
            files={p.relative_to(payload).as_posix():pin(p) for p in payload.rglob('*') if p.is_file()},
            scope='Four fresh unprofiled processes; complete prepared graph calls in both roles; fixed geometry iterations and original speed/repeatability gates; no application/ORT timing')
        save(payload/'payload.json',manifest)
        files={p.relative_to(ROOT).as_posix():pin(p) for p in [*TOOLS.iterdir(),MONITOR,BASE/'inputs.json',BASE/'selftest.json',LOCAL/'closed.json',AMD/'closed.json',PREVIOUS/'closed.json',CODEGEN/'closed.json'] if p.is_file()}
        for p in TOOLS.glob('*.py'):ast.parse(p.read_text(),str(p))
        with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
            for p in sorted(payload.rglob('*')):
                if p.is_file():archive.add(p,arcname=p.relative_to(payload).as_posix(),recursive=False)
        save(BASE/'prepared.json',dict(passed=True,files=files,payload=pin(payload/'payload.json'),archive=pin(BASE/'payload.tar.gz')))
        state['code']=0;print(json.dumps(dict(payload=pin(payload/'payload.json'),archive=pin(BASE/'payload.tar.gz'),jobs=JOBS,driver=diagnostic['driver'])))
    except BaseException:
        state.update(code=1,error=traceback.format_exc());raise
    finally:
        state['complete']=True;save(state_path,state)


if __name__=='__main__':prepare()
