"""Build the ordinary isolated LSTM candidate and qualify all captured recurrent outputs."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import traceback
import xml.etree.ElementTree as ET
from transform import PANELS, RECURRENT, HELPER, once, transform

ROOT=Path(__file__).resolve().parents[3]; TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/pyannote-lstm-input-blocks-20260922'
CONTROL=ROOT/'artifacts/pyannote-blocked-spatial-composition-v3-20260922/runtime'
FIXTURES=ROOT/'artifacts/pyannote-lstm-input-fixtures-v3-20260922'
ROOT_PROOF=ROOT/'artifacts/pyannote-blocked-spatial-root-20260922'
FEED=ROOT/'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
MONITOR=ROOT/'tests/parakeet/packing-budgets/common.py'
CORE='3c2f16b08856426d3dfeff07f1638dd76cee7f06b65bbee230e8e0789679206f'
spec=importlib.util.spec_from_file_location('lstm_blocks_monitor',MONITOR)
monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor);monitor.BASE=BASE
pin,read,save,verify,terminal=monitor.pin,monitor.read,monitor.save,monitor.verify,monitor.terminal


def inventory():
    value=read(BASE/'instructions.json');assert value['inventory_complete']
    core,data=value['observations'];assert core['assembly']=='Lokad.Onnx.dll' and data['assembly']=='Lokad.Onnx.Data.dll'
    for row in [core,data]:
        assert row['public_surface_equal'] and not row['removed']
        assert row['before_sha256']==pin(CONTROL/row['assembly'])['sha256'] and row['after_sha256']==pin(BASE/'runtime'/row['assembly'])['sha256']
    changed,=core['differences'];assert changed.startswith('Lokad.Onnx.CPUExecutionProvider::Lstm::')
    assert core['methods']==3161 and core['unchanged_methods']==3160
    assert len(core['added'])==2 and {k.split('::')[1] for k in core['added']}=={'InputBlock','LstmProjectOrderedRows'}
    assert data['methods']==data['unchanged_methods']==697 and not data['added'] and not data['differences']
    return dict(passed=True,changed=core['differences'],added=core['added'],unchanged_core=3160,unchanged_data=697,public_surface_equal=True)


def suite(name):
    p=BASE/'test-results'/(name+'.trx');doc=ET.parse(p);rows=doc.findall('.//{*}UnitTestResult');counts=doc.find('.//{*}Counters').attrib
    assert len(rows)==int(counts['total'])==int(counts['passed'])==149 and int(counts['failed'])==0
    assert all(r.attrib['outcome']=='Passed' for r in rows)
    assert sum('LstmInputBlockTests.' in r.attrib['testName'] for r in rows)==36
    return dict(passed=True,tests=149,new_tests=36,trx=pin(p))


def models():
    results={};reference=read(FIXTURES/'native/result.json');capture=read(FIXTURES/'output/result.json')
    for role in ['selected','candidate']:
        for width in ['256','scalar']:
            v=read(BASE/'output'/(role+'-'+width+'.json'))
            assert v['passed'] and v['role']==role and v['width']==width and v['core']==pin(BASE/('selected-runtime' if role=='selected' else 'runtime')/'Lokad.Onnx.dll')['sha256']
            assert v['calls']==24 and v['distinct_calls']==12 and v['outputs']==72 and v['values']==3631104
            assert v['readonly_operands'] and v['held_outputs_unchanged'] and v['no_performance_measurement'] and not v['avx512']
            assert v['hardware_accelerated']==(width=='256') and v['flags']==([] if width=='256' else ['DOTNET_EnableHWIntrinsic'])
            assert v['vector_count']==(8 if width=='256' else 4) and v['runtime']=='10.0.12'
            assert len(v['observations'])==72
            for i,row in enumerate(v['observations']):
                call=capture['calls'][i//6];slot=i%3;repeat=i//3%2;native=reference['reports'][i//6*3+slot]
                assert (row['name'],row['index'],row['repeat'],row['slot'])==(call['name'],call['index'],repeat,slot)
                assert row['exact'] and row['sha256']==call['outputs'][slot]['sha256'] and row['values']==call['outputs'][slot]['values']
                assert row['native_maximum']==native['comparison']['maximum']<=1e-4
                expected=0 if width=='scalar' else call['inputs'][1]['bytes']+call['inputs'][2]['bytes']+(8192 if role=='candidate' else 0)
                assert row['scratch_bytes']==expected
            results[role+'-'+width]={k:v[k] for k in ['passed','calls','outputs','values','maximum','vector_count']}
    return results


def main():
    assert not BASE.exists()
    assert pin(FIXTURES/'closed.json')['sha256']=='c64da99fa9e6cbae1af930b6560096c4d9fdd3e7efdf6c07fb6c4e0e53184f7b'
    proof=read(FIXTURES/'closed.json');assert proof['passed']
    for name,wanted in proof['files'].items():assert pin(FIXTURES/name)==wanted,name
    for identity in proof['identities']:terminal(identity)
    assert pin(CONTROL/'Lokad.Onnx.dll')['sha256']==CORE
    head=subprocess.check_output(['git','rev-parse','9533cd67'],cwd=ROOT,text=True).strip()
    BASE.mkdir();(BASE/'logs').mkdir();(BASE/'output').mkdir();source=BASE/'source';source.mkdir()
    archive=BASE/'source.tar'
    subprocess.run(['git','archive','--format=tar','--output',str(archive),head,'--','src','tests/Lokad.Onnx.Backend.Tests','tests/Shared','global.json','LICENSE.txt','icon.png','README.md','CHANGELOG.md'],cwd=ROOT,check=True)
    with tarfile.open(archive) as tar:tar.extractall(source,filter='data')
    texts,patch=transform((source/PANELS).read_text(),(source/RECURRENT).read_text())
    for name,text in texts.items():(source/name).write_text(text,encoding='utf8')
    (BASE/'candidate.patch').write_text(patch,encoding='utf8')
    target=source/'tests/Lokad.Onnx.Backend.Tests/LstmInputBlockTests.cs';shutil.copy2(TOOLS/'LstmInputBlockTests.cs.txt',target)
    old=source/'tests/Lokad.Onnx.Backend.Tests/LstmOutputLaneTests.cs';before=old.read_text()
    after=once(before,'3L * (w.Length + r.Length) * sizeof(float)','3L * (w.Length + r.Length + 16 * hidden) * sizeof(float)')
    old.write_text(after,encoding='utf8')
    (BASE/'scratch-test-before.cs.txt').write_text(before,encoding='utf8')
    consumer=BASE/'consumer';consumer.mkdir();shutil.copy2(TOOLS/'ModelReplay.cs',consumer/'ModelReplay.cs')
    project=consumer/'ModelReplay.csproj'
    project.write_text('''<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><AssemblyName>LstmModelReplay</AssemblyName><ImplicitUsings>enable</ImplicitUsings><Nullable>enable</Nullable></PropertyGroup><ItemGroup><Reference Include="Lokad.Onnx"><HintPath>../selected-runtime/Lokad.Onnx.dll</HintPath></Reference><Reference Include="Google.Protobuf"><HintPath>../selected-runtime/Google.Protobuf.dll</HintPath></Reference></ItemGroup></Project>''',encoding='utf8')
    shutil.copytree(CONTROL,BASE/'selected-runtime');shutil.copy2(ROOT/'PLAN.md',BASE/'prospective-plan.md')
    bridge=ROOT_PROOF/'bridge/bin/Release/net10.0/Bridge.dll'
    files={p.as_posix():pin(p) for folder in [source,TOOLS,consumer,BASE/'selected-runtime',bridge.parent] for p in folder.rglob('*') if p.is_file()}
    for p in [archive,BASE/'candidate.patch',BASE/'scratch-test-before.cs.txt',BASE/'prospective-plan.md',MONITOR,ROOT/'tests/parakeet/portable-models/common.py',FIXTURES/'closed.json',ROOT_PROOF/'closed.json']:files[p.as_posix()]=pin(p)
    for folder in [FIXTURES/'output',FIXTURES/'native']:
        for p in folder.iterdir():
            if p.is_file():files[p.as_posix()]=pin(p)
    save(BASE/'inputs.json',dict(files=files,source_commit=head,expected_methods=dict(changed=['CPUExecutionProvider.Lstm'],added=['LstmProjectionPanels.InputBlock','CPUExecutionProvider.LstmProjectOrderedRows']),no_performance_measurement=True))
    own=monitor.psutil.Process();state=dict(complete=False,code=None,supervisor=dict(pid=own.pid,birth=own.create_time()),runs=[])
    path=BASE/'controller.json';save(path,state);jobs={};flags=monitor.FLAGS+['-p:NuGetAudit=false']
    def run(name,args,numerical=False,children=False,environment=None):
        jobs[name]=[12 if numerical else 8,8,900,numerical and not children];save(BASE/'jobs.json',jobs)
        original=monitor.clean_env
        if environment:monitor.clean_env=lambda:original()|environment
        output=BASE/'test-results' if children else BASE/'output' if numerical else BASE
        try:monitor.worker(state,path,name,args,ROOT,[0],jobs[name][0],8,900,children or not numerical,output)
        finally:monitor.clean_env=original
        print(name,'passed',flush=True)
    try:
        cli=source/'src/Lokad.Onnx.CLI/Lokad.Onnx.CLI.csproj';backend=source/'tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj'
        for name,p in [('cli',cli),('backend',backend),('consumer',project)]:
            run(name+'-restore',['dotnet','restore',p,*flags,'--source',FEED,'--packages',BASE/'packages'])
            run(name+'-build',['dotnet','build',p,'-c','Release',*flags,'--no-restore','--disable-build-servers'])
        shutil.copytree(cli.parent/'bin/Release/net10.0',BASE/'runtime')
        for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']:assert pin(backend.parent/'bin/Release/net10.0'/name)==pin(BASE/'runtime'/name)
        run('inventory',['dotnet',bridge,CONTROL,BASE/'runtime',BASE/'instructions.json'])
        save(BASE/'instruction-review.json',inventory())
        for role in ['selected','candidate']:
            folder=BASE/('selected-runtime' if role=='selected' else 'runtime')
            for suffix in ['dll','deps.json','runtimeconfig.json']:shutil.copy2(consumer/'bin/Release/net10.0'/('LstmModelReplay.'+suffix),folder/('LstmModelReplay.'+suffix))
        binaries={p.as_posix():pin(p) for folder in [BASE/'runtime',BASE/'selected-runtime',backend.parent/'bin/Release/net10.0'] for p in folder.rglob('*') if p.is_file()}
        save(BASE/'binaries.json',dict(files=binaries))
        tests={}
        for mode in ['ordinary','scalar']:
            name='lstm-'+mode
            run(name,['dotnet','test',backend,'-c','Release',*flags,'--no-build','--no-restore','--filter','FullyQualifiedName~Lstm',
                '--logger','trx;LogFileName='+name+'.trx','--results-directory',BASE/'test-results'],True,True,{'DOTNET_EnableHWIntrinsic':'0'} if mode=='scalar' else None)
            tests[name]=suite(name)
        for role in ['selected','candidate']:
            folder=BASE/('selected-runtime' if role=='selected' else 'runtime');core=pin(folder/'Lokad.Onnx.dll')['sha256']
            for width in ['256','scalar']:
                run(role+'-'+width,['dotnet',folder/'LstmModelReplay.dll',FIXTURES,BASE/'output'/(role+'-'+width+'.json'),core,role,width],True,
                    environment={'DOTNET_EnableHWIntrinsic':'0'} if width=='scalar' else None)
        verify(files);verify(binaries)
        save(BASE/'verified.json',dict(passed=True,core=pin(BASE/'runtime/Lokad.Onnx.dll'),data=pin(BASE/'runtime/Lokad.Onnx.Data.dll'),
            inventory=inventory(),tests=tests,models=models(),no_performance_measurement=True,actual_amd_pending=True))
        state['code']=0
    except BaseException:
        state.update(code=1,error=traceback.format_exc());raise
    finally:state['complete']=True;save(path,state)


if __name__=='__main__':main()
