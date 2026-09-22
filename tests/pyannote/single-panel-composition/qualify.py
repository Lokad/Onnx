"""Run focused/full suites and consume the actual isolated NuGet package."""
import collections
import importlib.util
import json
import shutil
import traceback
import xml.etree.ElementTree as ET
import zipfile
from prepare import *

SUITES=[('focused','FullyQualifiedName~ConvDirectOutputTests',False,23,0),
    ('hardware-disabled','FullyQualifiedName~ConvDirectOutputTests',True,23,0),
    ('backend-full',None,False,3313,93),('tensors-full',None,False,343,0)]


def suite(name,passed,skipped):
    path=BASE / 'test-results' / (name+'.trx');tree=ET.parse(path)
    counts=collections.Counter(x.attrib['outcome'] for x in tree.findall('.//{*}UnitTestResult'))
    assert counts==dict(Passed=passed,**({'NotExecuted':skipped} if skipped else {})),counts
    counters=tree.find('.//{*}Counters').attrib
    assert int(counters['passed'])==int(counters['executed'])==passed
    assert int(counters['total'])==passed+skipped and int(counters['failed'])==0
    return dict(name=name,passed=passed,skipped=skipped,counters=counters,outcomes=dict(counts),trx=pin(path))


def main():
    prepared=read(BASE / 'prepared.json');assert prepared['passed'];verify(prepared['files'])
    old=read(BASE / 'preparation.json');assert old['complete'] and old['code']==0;terminal(old['supervisor'])
    for row in old['runs']:
        assert row['complete'] and row['code']==0
        for pid,birth in row['members'].items():terminal(dict(pid=int(pid),birth=birth))
    assert not (BASE / 'qualification.json').exists()
    review();source=BASE / 'source';runtime=BASE / 'runtime'
    own=psutil.Process();state=dict(complete=False,code=None,supervisor=dict(pid=own.pid,birth=own.create_time()),runs=[])
    path=BASE / 'qualification.json';save(path,state);flags=monitor.FLAGS+['-p:NuGetAudit=false']
    files={rel(p):pin(p) for p in [TOOLS / 'qualify.py',BASE / 'prepared.json',BASE / 'instruction-review.json']}
    probe=ROOT / 'tests/pyannote/portable-integration/PackageProbeV2.cs';files[rel(probe)]=pin(probe)
    save(BASE / 'qualification-inputs.json',dict(passed=True,files=files,suites=SUITES))
    def run(name,command,inference,output):
        monitor.worker(state,path,name,command,source,[0],10 if inference else 8,8,900,True,output);print(name,'passed',flush=True)
    suites=[]
    try:
        for name,pattern,disabled,passed,skipped in SUITES:
            project=source / ('tests/Lokad.Onnx.Tensors.Tests/Lokad.Onnx.Tensors.Tests.csproj' if name=='tensors-full' else 'tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj')
            command=['dotnet','test',project,'-c','Release',*flags,'--no-build','--no-restore',
                '--logger','trx;LogFileName='+name+'.trx','--results-directory',BASE / 'test-results']
            if pattern:command+=['--filter',pattern]
            old_env=monitor.clean_env
            if disabled:monitor.clean_env=lambda:old_env() | {'DOTNET_EnableHWIntrinsic':'0'}
            try:run(name,command,True,BASE / 'test-results')
            finally:monitor.clean_env=old_env
            suites.append(suite(name,passed,skipped));save(BASE / 'suites.json',suites)
        run('package',['dotnet','pack',source / 'src/Lokad.Onnx/Lokad.Onnx.csproj','-c','Release',*flags,
            '--no-build','--no-restore','--output',BASE / 'nuget'],False,BASE / 'nuget')
        package=BASE / 'nuget/Lokad.Onnx.0.2.0.nupkg'
        with zipfile.ZipFile(package) as archive:
            assert archive.read('lib/net10.0/Lokad.Onnx.dll')==(runtime / 'Lokad.Onnx.dll').read_bytes()
            document=ET.fromstring(archive.read('Lokad.Onnx.nuspec'))
            dependencies=[n.attrib for n in document.iter() if n.tag.split('}')[-1]=='dependency']
            assert len(dependencies)==1 and dependencies[0]['id']=='Google.Protobuf' and dependencies[0]['version']=='3.33.5'
            save(BASE / 'package.json',dict(passed=True,package=pin(package),entries=archive.namelist(),dependencies=dependencies,core=pin(runtime / 'Lokad.Onnx.dll')))
        consumer=BASE / 'package-consumer';consumer.mkdir()
        project=consumer / 'PackageProbe.csproj'
        project.write_text('<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework></PropertyGroup><ItemGroup><PackageReference Include="Lokad.Onnx" Version="0.2.0" /></ItemGroup></Project>\n',encoding='utf8')
        program=probe.read_text(encoding='utf8')
        extra='''// Public tiled convolution must execute the new admitted route from the package.
var tileInput = new DenseTensor<float>(Enumerable.Repeat(.125f, 64 * 1040).ToArray(), new int[] { 1, 64, 1, 1040 });
var tileWeight = new DenseTensor<float>(Enumerable.Repeat(1f, 32 * 64 * 3).ToArray(), new int[] { 32, 64, 1, 3 });
var tileBias = new DenseTensor<float>(Enumerable.Repeat(1f, 32).ToArray(), new int[] { 32 });
var tileResult = Tensor<float>.Conv2D(tileInput, tileWeight, 1, new int[] { 0, 0, 0, 0 }, tileBias,
    new int[] { 1, 3 }, new int[] { 1, 1 }, new int[] { 1, 1 }, TensorExecutionOptions.Auto);
Require(tileResult.Length == 32 * 1038 && tileResult.ToArray().All(v => v == 25f), "Packaged tiled convolution differs");
Require(tileInput.ToArray().All(v => v == .125f) && tileWeight.ToArray().All(v => v == 1f), "Packaged tiled inputs changed");
// The public planner gives this matrix a 32-column full tile and an 8-column tail.
var narrowInput = new DenseTensor<float>(Enumerable.Repeat(.125f, 256 * 3 * 42).ToArray(), new int[] { 1, 256, 3, 42 });
var narrowWeight = new DenseTensor<float>(Enumerable.Repeat(1f, 256 * 256 * 3 * 3).ToArray(), new int[] { 256, 256, 3, 3 });
var narrowBias = new DenseTensor<float>(Enumerable.Repeat(1f, 256).ToArray(), new int[] { 256 });
var scratch = new ScratchAccountant();
var narrowResult = Tensor<float>.Conv2D(narrowInput, narrowWeight, 1, new int[] { 0, 0, 0, 0 }, narrowBias,
    new int[] { 3, 3 }, new int[] { 1, 1 }, new int[] { 1, 1 }, TensorExecutionOptions.Auto with { ScratchReporter = scratch });
Require(narrowResult.Length == 256 * 40 && narrowResult.ToArray().All(v => v == 289f), "Packaged narrow convolution differs");
Require(scratch.TotalScratchBytes == (2304L + 256) * 32 * sizeof(float), "Single-panel path still rents packed scratch");
Require(narrowInput.ToArray().All(v => v == .125f) && narrowWeight.ToArray().All(v => v == 1f), "Packaged narrow inputs changed");
'''
        assert program.count('var result = new {')==1
        program=program.replace('var result = new {',extra+'var result = new { packaged_tiled_convolution_values = tileResult.Length, packaged_narrow_values = narrowResult.Length, narrow_scratch_bytes = scratch.TotalScratchBytes,')
        (consumer / 'Program.cs').write_text(program,encoding='utf8')
        run('consumer-restore',['dotnet','restore',project,*flags,'--source',BASE / 'nuget','--source',FEED,'--packages',BASE / 'consumer-cache'],False,None)
        run('consumer-build',['dotnet','build',project,'-c','Release',*flags,'--no-restore','--disable-build-servers'],False,consumer)
        assert pin(consumer / 'bin/Release/net10.0/Lokad.Onnx.dll')==pin(runtime / 'Lokad.Onnx.dll')
        run('consumer',['dotnet',consumer / 'bin/Release/net10.0/PackageProbe.dll',source / 'tests/Lokad.Onnx.Backend.Tests/models/mnist-8.onnx',
            pin(runtime / 'Lokad.Onnx.dll')['sha256'],BASE / 'consumer.json'],True,consumer)
        result=read(BASE / 'consumer.json');assert result['passed'] and result['packaged_tiled_convolution_values']==32*1038 and result['packaged_narrow_values']==10240 and result['narrow_scratch_bytes']==327680
        verify(files);verify(prepared['files'])
        for folder in [consumer,BASE / 'nuget']:
            files.update({rel(p):pin(p) for p in folder.rglob('*') if p.is_file() and 'obj' not in p.relative_to(folder).parts})
        save(BASE / 'qualified.json',dict(passed=True,files=files,suites=suites,core=prepared['core'],data=prepared['data'],package=pin(package),
            production_changed=False,model_qualified=False,performance_qualified=False))
        state['code']=0
    except BaseException:
        state.update(code=1,error=traceback.format_exc());raise
    finally:
        state['complete']=True;save(path,state)


if __name__=='__main__':main()
