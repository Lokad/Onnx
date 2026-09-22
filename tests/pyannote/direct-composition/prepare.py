"""Normal isolated source composition with exhaustive instruction and caller checks."""
import difflib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import traceback

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
BASE=ROOT / 'artifacts/pyannote-direct-composition-20260922'
CONTROL=ROOT / 'artifacts/pyannote-portable-applications-20260922/runtime'
COMPONENT=ROOT / 'artifacts/pyannote-two-column-20260922'
FEED=ROOT / 'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
MONITOR=ROOT / 'tests/parakeet/packing-budgets/common.py'
spec=importlib.util.spec_from_file_location('direct_composition_monitor',MONITOR)
monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor);monitor.BASE=BASE
pin,read,save,verify,terminal,psutil=monitor.pin,monitor.read,monitor.save,monitor.verify,monitor.terminal,monitor.psutil
sys.path.insert(0,str(TOOLS.parent / 'two-column'))
from two_column_generator import generate


def rel(p):return p.relative_to(ROOT).as_posix()


def review():
    inventory=read(BASE / 'instructions.json');assert inventory['inventory_complete']
    assert [(r['assembly'],r['methods']) for r in inventory['observations']]==[('Lokad.Onnx.dll',3108),('Lokad.Onnx.Data.dll',697)]
    for row in inventory['observations']:
        assert row['public_surface_equal'] and not row['removed']
        assert row['before_sha256']==pin(CONTROL / row['assembly'])['sha256']
        assert row['after_sha256']==pin(BASE / 'runtime' / row['assembly'])['sha256']
        if row['assembly']=='Lokad.Onnx.Data.dll':
            assert row['unchanged_methods']==697 and not row['added'] and not row['differences'];continue
        assert row['unchanged_methods']==3107 and len(row['differences'])==1
        assert row['differences'][0].startswith('Lokad.Onnx.Tensor`1[T]::RunTiledBatchFloat::')
        assert len(row['added'])==5 and {k.split('::')[1] for k in row['added']}=={
            'TryConvDirectOutput','Multiply','MultiplyTwoColumns','AddBiasVector','AddBiasScalar'}
        prototype={k:v for k,v in inventory['component_methods'].items() if k.startswith('DirectOutput::')}
        assert len(prototype)==4
        for key,value in prototype.items():
            new_key=key.replace('DirectOutput::','Lokad.Onnx.ConvDirectOutput::')
            assert row['candidate_methods'][new_key]==value.replace('DirectOutput::','Lokad.Onnx.ConvDirectOutput::'),key
        caller=json.loads(row['candidate_methods'][row['differences'][0]])
        calls=[i['operand'] for i in caller['instructions'] if i['opcode'] in ['call','callvirt','newobj']]
        assert next(i for i,s in enumerate(calls) if 'TryConvDirectOutput(' in s)<next(i for i,s in enumerate(calls) if 'TryConvPortableRows(' in s)<next(i for i,s in enumerate(calls) if 'MatMul2D(' in s)
    return dict(passed=True,compiled_kernel_bodies_equal=True,changed_existing_core_methods=1,
        added_core_methods=5,unchanged_core_methods=3107,unchanged_data_methods=697,instructions=pin(BASE / 'instructions.json'))


def main():
    assert not BASE.exists()
    assert pin(COMPONENT / 'closed.json')['sha256']=='848b9681e1914818a5a9a0a63e98f22b7d1abf06dbc906d243d0527094df2e39'
    closed=read(COMPONENT / 'closed.json');assert closed['passed'];verify(closed['files'])
    assert read(COMPONENT / 'analysis.json')['eligible']
    assert pin(CONTROL / 'Lokad.Onnx.dll')['sha256']=='e9c87932b2184c2f6bfef72faabb1719bdbceadc779a15fe1ffd3f3056d02838'
    assert pin(CONTROL / 'Lokad.Onnx.Data.dll')['sha256']=='85d166b59e2beef18ca7664f76faf445bf3cd81509f8f1d1c4b3c5354f53757a'
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert head.startswith('8b467dc9')
    BASE.mkdir();(BASE / 'logs').mkdir()
    archive=BASE / 'source.tar'
    subprocess.run(['git','archive','--format=tar','--output',str(archive),head],cwd=ROOT,check=True)
    source=BASE / 'source';source.mkdir()
    with tarfile.open(archive) as tar:tar.extractall(source,filter='data')
    kernel,original=generate((source / 'src/Lokad.Onnx/MathOps.cs').read_text(encoding='utf-8-sig'))
    assert kernel==(COMPONENT / 'normal/DirectOutput.cs').read_text(encoding='utf8')
    assert kernel.count('static class DirectOutput')==1
    product=kernel.replace('static class DirectOutput','namespace Lokad.Onnx;\n\ninternal static class ConvDirectOutput')
    (source / 'src/Lokad.Onnx/Zzz.ConvDirectOutputKernel.cs').write_text(product,encoding='utf8')
    shutil.copy2(TOOLS / 'ConvDirectOutput.cs',source / 'src/Lokad.Onnx/Zzz.ConvDirectOutput.cs')
    shutil.copy2(TOOLS / 'ConvDirectOutputTests.cs',source / 'tests/Lokad.Onnx.Backend.Tests/ConvDirectOutputTests.cs')
    path=source / 'src/Lokad.Onnx/TensorOps.ConvPool.cs'; before=path.read_text(encoding='utf-8-sig')
    old='''                if (!TryConvPortableRows(wView.Buffer.Span, pView.Buffer.Span, dView.Buffer.Span,
                    tileM, tileKg, colCount, options))
                    Tensor<float>.MatMul2D(wView, pView, dView, options);
                int outBase = b * outBatch + g * tileM * tileN;'''
    new='''                int outBase = b * outBatch + g * tileM * tileN;
                if (TryConvDirectOutput(wView.Buffer.Span, pView.Buffer.Span, dView.Buffer.Span,
                    os.Slice(outBase + colStart), hasBias ? bs.Slice(g * tileM, tileM) : default,
                    hasBias, tileM, tileKg, colCount, tileN, options)) continue;
                if (!TryConvPortableRows(wView.Buffer.Span, pView.Buffer.Span, dView.Buffer.Span,
                    tileM, tileKg, colCount, options))
                    Tensor<float>.MatMul2D(wView, pView, dView, options);'''
    assert before.count(old)==1;after=before.replace(old,new);path.write_text(after,encoding='utf8')
    patch=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/src/Lokad.Onnx/TensorOps.ConvPool.cs',tofile='b/src/Lokad.Onnx/TensorOps.ConvPool.cs'))
    for name in ['src/Lokad.Onnx/Zzz.ConvDirectOutput.cs','src/Lokad.Onnx/Zzz.ConvDirectOutputKernel.cs','tests/Lokad.Onnx.Backend.Tests/ConvDirectOutputTests.cs']:
        patch+=''.join(difflib.unified_diff([], (source/name).read_text(encoding='utf8').splitlines(True),fromfile='/dev/null',tofile='b/'+name))
    (BASE / 'candidate.patch').write_text(patch,encoding='utf8')
    bridge=BASE / 'bridge';bridge.mkdir()
    inventory_source=(TOOLS.parent / 'combined-avx512/Inventory.cs.txt').read_text(encoding='utf8')
    assert inventory_source.count('args.Length != 3')==1
    inventory_source=inventory_source.replace('args.Length != 3','args.Length != 4')
    old='observations }, new JsonSerializerOptions';new='observations, component_methods = Inspect(Load(Path.GetFullPath(args[3]), "DirectOutputProbe.dll", "component")) }, new JsonSerializerOptions'
    assert inventory_source.count(old)==1;inventory_source=inventory_source.replace(old,new)
    (bridge / 'Program.cs').write_text(inventory_source,encoding='utf8')
    shutil.copy2(ROOT / 'artifacts/pyannote-portable-integration-20260922/bridge/Bridge.csproj',bridge / 'Bridge.csproj')
    probe=BASE / 'caller';probe.mkdir();shutil.copy2(TOOLS / 'CallerProbe.cs',probe / 'Program.cs')
    (probe / 'Caller.csproj').write_text(f'''<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><Nullable>enable</Nullable></PropertyGroup><ItemGroup><Reference Include="Lokad.Onnx"><HintPath>{BASE / 'runtime/Lokad.Onnx.dll'}</HintPath></Reference></ItemGroup></Project>''',encoding='utf8')
    files={rel(p):pin(p) for folder in [source,bridge,probe,TOOLS] for p in folder.rglob('*') if p.is_file()}
    for p in [MONITOR,archive,COMPONENT / 'closed.json',COMPONENT / 'payload/shapes.json',BASE / 'candidate.patch',
        TOOLS.parent / 'combined-avx512/Inventory.cs.txt',*CONTROL.glob('*.dll'),COMPONENT / 'payload/runtime/DirectOutputProbe.dll']:
        files[rel(p)]=pin(p)
    save(BASE / 'inputs.json',dict(passed=True,files=files,source_commit=head,production_changed=False))
    own=psutil.Process();state=dict(complete=False,code=None,supervisor=dict(pid=own.pid,birth=own.create_time()),runs=[])
    path=BASE / 'preparation.json';save(path,state);flags=monitor.FLAGS+['-p:NuGetAudit=false']
    def run(name,command,output):
        monitor.worker(state,path,name,command,source,[0],8,8,900,True,output);print(name,'passed',flush=True)
    def build(name,project):
        run(name+'-restore',['dotnet','restore',project,*flags,'--source',FEED,'--packages',BASE / 'packages'],None)
        run(name+'-build',['dotnet','build',project,'-c','Release',*flags,'--no-restore','--disable-build-servers'],None)
    try:
        for name,project in [('cli',source / 'src/Lokad.Onnx.CLI/Lokad.Onnx.CLI.csproj'),
            ('backend',source / 'tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj'),
            ('tensors',source / 'tests/Lokad.Onnx.Tensors.Tests/Lokad.Onnx.Tensors.Tests.csproj'),('bridge',bridge / 'Bridge.csproj')]:
            build(name,project)
        shutil.copytree(source / 'src/Lokad.Onnx.CLI/bin/Release/net10.0',BASE / 'runtime')
        for name in ['Backend','Tensors']:
            assert pin(source / f'tests/Lokad.Onnx.{name}.Tests/bin/Release/net10.0/Lokad.Onnx.dll')==pin(BASE / 'runtime/Lokad.Onnx.dll')
        run('inventory',['dotnet',bridge / 'bin/Release/net10.0/Bridge.dll',CONTROL,BASE / 'runtime',BASE / 'instructions.json',COMPONENT / 'payload/runtime'],bridge)
        save(BASE / 'instruction-review.json',review())
        build('caller',probe / 'Caller.csproj')
        assert pin(probe / 'bin/Release/net10.0/Lokad.Onnx.dll')==pin(BASE / 'runtime/Lokad.Onnx.dll')
        for mode in ['normal','disabled']:
            old_env=monitor.clean_env
            if mode=='disabled':monitor.clean_env=lambda:old_env() | {'DOTNET_EnableHWIntrinsic':'0'}
            try:
                run('caller-'+mode,['dotnet',probe / 'bin/Release/net10.0/Caller.dll',CONTROL,COMPONENT / 'payload/shapes.json',
                    BASE / ('caller-'+mode+'.json'),mode,pin(BASE / 'runtime/Lokad.Onnx.dll')['sha256']],probe)
            finally:monitor.clean_env=old_env
            result=read(BASE / ('caller-'+mode+'.json'));assert result['passed'] and len(result['records'])==736
        verify(files)
        for folder in [source,bridge,probe,BASE / 'runtime']:
            files.update({rel(p):pin(p) for p in folder.rglob('*') if p.is_file() and 'obj' not in p.relative_to(folder).parts})
        save(BASE / 'prepared.json',dict(passed=True,files=files,core=pin(BASE / 'runtime/Lokad.Onnx.dll'),data=pin(BASE / 'runtime/Lokad.Onnx.Data.dll'),
            caller_cases_per_mode=736,source_commit=head,production_changed=False,models_qualified=False,performance_qualified=False))
        state['code']=0;print('Normal direct-output composition prepared',flush=True)
    except BaseException:
        state.update(code=1,error=traceback.format_exc());raise
    finally:
        state['complete']=True;save(path,state)


if __name__=='__main__':main()
