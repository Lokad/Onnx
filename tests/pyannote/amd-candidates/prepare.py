"""Build an offline AMD qualification payload without accessing the VM."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import zipfile

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/pyannote-amd-candidates-20260921'
ROWS=ROOT/'artifacts/pyannote-conv-row-sharing-v2-20260921'
PORTABLE=ROOT/'artifacts/pyannote-lstm-output-lanes-20260921'
PRODUCT=ROOT/'artifacts/whisper-memory-product-v2-20260921/source/src/Lokad.Onnx.CLI/bin/Release/net10.0'
PRIOR=ROOT/'artifacts/audio-amd-two-family-20260920'
PARAKEET=ROOT/'artifacts/parakeet-transcription-20260919/frozen'
REMOTE='/dev/shm/lokad-pyannote-candidates-20260921'


def pin(path):
    with path.open('rb') as f:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())


def read(path):return json.loads(path.read_text())


def save(path,value):
    with path.open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def copy(source,target):target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)


def clean_env():return {k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}


def specs(value):
    if isinstance(value,dict):
        if set(['path','bytes','sha256'])<=set(value):yield value
        else:
            for item in value.values():yield from specs(item)
    elif isinstance(value,list):
        for item in value:yield from specs(item)


def main():
    prepared=read(ROWS/'prepared.json');local=read(ROWS/'local-preparation.json')
    assert local['local_preparation_passed'] and not local['avx512_execution_qualified']
    assert pin(ROWS/'prepared.json')['sha256']=='724630645d35446464ce298c182b3bd7f2ff79d9e6d01e50a4ebe540bde09f1a'
    for name,wanted in prepared['files'].items():assert pin(ROOT/name)==wanted,name
    closed=read(PORTABLE/'qualification-closed.json');assert closed['passed']
    assert pin(PORTABLE/'qualification-closed.json')['sha256']=='cf950ec5cedf702c1af38decc377cd516a5d8b652f77853d03d2c81db0b5bf53'
    for name,wanted in closed['files'].items():assert pin(ROOT/name)==wanted,name
    prior_closed=read(PRIOR/'closed.json');assert prior_closed['passed']
    for name,wanted in prior_closed['files'].items():assert pin(ROOT/name)==wanted,name
    old=PRIOR/'collected';frozen=read(old/'frozen.json')
    assert pin(old/'frozen.json')==pin(PRIOR/'frozen.json')
    BASE.mkdir(exist_ok=False);payload=BASE/'payload';payload.mkdir();(BASE/'logs').mkdir()
    source=payload/'source';shutil.copytree(ROWS/'candidate-source',source,ignore=shutil.ignore_patterns('bin','obj'))
    assert (source/'Lokad.Onnx.slnx').exists()
    source_files={p.relative_to(source).as_posix():pin(p) for p in source.rglob('*') if p.is_file()}
    assets=read(ROWS/'candidate-source/tests/Lokad.Onnx.Backend.Tests/obj/project.assets.json')
    packages={};unpacked=0
    for library in assets['libraries'].values():
        if library['type']!='package':continue
        filename=library['path'].replace('/','.')+'.nupkg'
        path=next(p for folder in assets['packageFolders'] if (p:=Path(folder)/library['path']/filename).is_file())
        copy(path,payload/'nuget-feed'/filename);packages[filename]=pin(path)
        with zipfile.ZipFile(path) as archive:unpacked+=sum(info.file_size for info in archive.infolist())
    assert len(packages)==20 and sum(p['bytes'] for p in packages.values())==23601190
    cores={}
    for role,directory in [('production',PRODUCT),('portable',PORTABLE/'runtimes/candidate'),('rows',ROWS/'runtime')]:
        target=payload/'runtimes'/role;target.mkdir(parents=True)
        for path in PRODUCT.glob('*.dll'):copy(path,target/path.name)
        copy(directory/'Lokad.Onnx.dll',target/'Lokad.Onnx.dll')
        for source_dir,name in [(ROOT/'artifacts/audio-ort-baseline-v2-20260919/bin','AudioBenchmark'),(PARAKEET/'replay','TranscribeReplay')]:
            for suffix in ['.dll','.deps.json','.runtimeconfig.json']:copy(source_dir/(name+suffix),target/(name+suffix))
        cores[role]=pin(target/'Lokad.Onnx.dll')
        assert pin(target/'Lokad.Onnx.Data.dll')['sha256']=='e7fe1668e3aa08fb07b1e5a687ef2b1e4af54567f6a458db09d411eb69f99aeb'
        assert pin(target/'AudioBenchmark.dll')['sha256']=='7eca033a1b986a4cb90621392639d230c95097cb703dd25274fd72d66c5ba4f1'
        assert pin(target/'TranscribeReplay.dll')['sha256']=='335ca09d0e45e344068c484c92af9d0db43a6ae0accd1895d7ae7bb88b0afcf9'
    assert {k:v['sha256'] for k,v in cores.items()}==dict(production='d1f86a7346dcd70ebcc9ef7d9cd9633f05ad3a5275ca39f035c72325a0531fa4',
        portable='469cb2d6a4558d917266434bd1f968c8b9f2762795b963800b2d945083852edd',rows='29477d505dd230aef0b5aa2792da8ec76c6903c5d2cff244327d803b432b4cbb')
    external={name:entry for name,entry in frozen['external'].items() if '.so' in Path(name).name}
    for name in ['native.py','protocol.py','audio_adapter.py','campaign_processes.py']:
        path=old/'runtime'/name;assert pin(path)==frozen['files']['runtime/'+name];copy(path,payload/'runtime'/name)
    for family in ['pyannote','parakeet']:
        manifest=read(old/'manifests'/(family+'.json'))
        assert pin(old/'manifests'/(family+'.json'))==frozen['files']['manifests/'+family+'.json']
        for item in specs(manifest):
            name=item['path'];wanted={k:item[k] for k in ['bytes','sha256']}
            if name.startswith('/'):
                assert frozen['external'][name]==wanted;external[name]=wanted;continue
            origin=(old/'assets'/name).resolve();target=(payload/'assets'/name).resolve()
            assert target.is_relative_to(payload.resolve()) and origin.is_relative_to(old.resolve())
            assert pin(origin)==wanted;copy(origin,target)
        # Explicit role manifests keep native assets identical and bind the core
        # appropriate for each managed worker. Native role uses production's copy.
        for role in cores:
            target=payload/'manifests'/(role+'-'+family+'.json');target.parent.mkdir(exist_ok=True)
            value=dict(manifest,product_source='isolated-pyannote-'+role,core_sha256=cores[role]['sha256'],
                data_sha256=pin(payload/'runtimes'/role/'Lokad.Onnx.Data.dll')['sha256'])
            save(target,value)
    # Original full decoder trajectory arrays, not timing-only transcript fixtures.
    reference=read(PARAKEET/'reference/manifest.json')
    assert pin(PARAKEET/'reference/manifest.json')['sha256']=='3bad7d262b8809b1265c84c8e66d02ee38e7d4cff2d92014448976a9e161103c'
    copy(PARAKEET/'reference/manifest.json',payload/'parakeet-reference/manifest.json')
    for name,item in reference['files'].items():
        path=PARAKEET/'reference'/name;assert pin(path)=={k:item[k] for k in ['bytes','sha256']};copy(path,payload/'parakeet-reference'/name)
    # Six native graph outputs for the three pyannote crops, all values retained.
    original=read(ROOT/'artifacts/audio-ort-baseline-v2-20260919/inputs/pyannote.json')
    reference_path=ROOT/original['reference']['path'];reference=read(reference_path);graph_cases=[]
    assert pin(reference_path)=={k:original['reference'][k] for k in ['bytes','sha256']}
    for case in reference['cases']:
        if case['seconds']!=10:continue
        for model,key in [('segmentation','scores'),('embedding','encoded')]:
            name=case['windows'][0][key];path=reference_path.parent/name
            assert pin(path)=={k:reference['files'][name][k] for k in ['bytes','sha256']}
            target=payload/'graph-reference'/name;copy(path,target)
            graph_cases.append(dict(name=case['name'],model=model,path=target.relative_to(payload).as_posix(),**pin(target)))
    assert len(graph_cases)==6;save(payload/'graph-reference.json',graph_cases)
    consumer=payload/'graph-consumer';consumer.mkdir()
    original_program=ROOT/'artifacts/pyannote-spatial-panels-20260921/consumer/Program.cs'
    program=original_program.read_text();old_guard='if (!OperatingSystem.IsWindows()) throw new PlatformNotSupportedException("This local attribution uses Windows CPU2.");'
    new_guard='if (!OperatingSystem.IsWindows() && !OperatingSystem.IsLinux()) throw new PlatformNotSupportedException("Verified CPU2 affinity required.");'
    assert program.count(old_guard)==1
    (consumer/'Program.cs').write_text(program.replace(old_guard,new_guard),encoding='utf8')
    project=(ROOT/'tests/pyannote/performance-profile/Profile.csproj').read_text().replace('../../Shared/NpySupport.cs','NpySupport.cs')
    project=project.replace('<OutputType>Exe</OutputType>','<OutputType>Exe</OutputType><AssemblyName>GraphQualification</AssemblyName>')
    (consumer/'GraphQualification.csproj').write_text(project,encoding='utf8')
    copy(ROOT/'tests/Shared/NpySupport.cs',consumer/'NpySupport.cs')
    bridge=payload/'il-bridge';bridge.mkdir();copy(ROOT/'tests/whisper/memory-product-v2/CompareIlStable.cs',bridge/'Program.cs')
    (bridge/'IlBridge.csproj').write_text('<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><ImplicitUsings>enable</ImplicitUsings><Nullable>enable</Nullable></PropertyGroup></Project>\n')
    flags=['--tl:off','--nologo','-v','minimal','-p:EnableSourceControlManagerQueries=false','-p:EnableSourceLink=false']
    commands=[];packages_path=BASE/'local-packages';feed=payload/'nuget-feed'
    def command(name,args):
        with (BASE/'logs'/(name+'.log')).open('x') as log:
            code=subprocess.run(args,cwd=source,env=clean_env(),stdout=log,stderr=subprocess.STDOUT,timeout=900).returncode
        commands.append(dict(name=name,command=args,code=code))
        (BASE/'local-builds.json').write_text(json.dumps(commands,indent=2)+'\n')
        assert code==0,name;print(name,'passed',flush=True)
    for name,project in [('backend',source/'tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj'),
        ('tensors',source/'tests/Lokad.Onnx.Tensors.Tests/Lokad.Onnx.Tensors.Tests.csproj'),
        ('graph-consumer',consumer/'GraphQualification.csproj'),('il-bridge',bridge/'IlBridge.csproj')]:
        command(name+'-restore',['dotnet','restore',str(project),*flags,'--source',str(feed),'--packages',str(packages_path),'--no-http-cache','-p:NuGetAudit=false'])
        extra=['-p:FrozenProductDirectory='+str(payload/'runtimes/rows')] if name=='graph-consumer' else []
        command(name+'-build',['dotnet','build',str(project),'-c','Release',*flags,'--no-restore',*extra])
    command('il-bridge',['dotnet',str(bridge/'bin/Release/net10.0/IlBridge.dll'),str(payload/'runtimes/rows'),
        str(source/'tests/Lokad.Onnx.Backend.Tests/bin/Release/net10.0'),str(BASE/'logs/local-il-bridge.json')])
    assert read(BASE/'logs/local-il-bridge.json')['passed']
    for role in cores:
        for suffix in ['.dll','.deps.json','.runtimeconfig.json']:
            copy(consumer/'bin/Release/net10.0'/('GraphQualification'+suffix),payload/'runtimes'/role/('GraphQualification'+suffix))
    for name,wanted in source_files.items():assert pin(source/name)==wanted,name
    for path in TOOLS.glob('*.py'):copy(path,payload/'tools'/path.name)
    copy(ROOT/'.agent/m3-audio-amd-candidates-20260921.md',payload/'prospective-plan.md')
    files={p.relative_to(payload).as_posix():pin(p) for p in payload.rglob('*') if p.is_file() and not {'bin','obj'}.intersection(p.relative_to(payload).parts)}
    spec=dict(schema=1,source=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),files=files,
        cores=cores,source_files=source_files,external=external,python_paths=frozen['python_paths'],interpreter=frozen['interpreter'],
        remote=REMOTE,nuget=dict(archives=packages,uncompressed_bytes=unpacked),
        graph_consumer_adaptation=dict(original=pin(original_program),old_guard=old_guard,new_guard=new_guard),
        protocol=dict(qualification_roles=['production','portable','rows'],graph_arrays_per_role=18,parakeet_arrays_per_role=784,
            parakeet_values_per_role=3090494,timing_roles=['production','portable','rows','ort','ort','rows','portable','production'],
            timing_cases=4,warmup_passes=1,measured_passes=3,total_timing_calls=128,measured_timing_calls=96),
        scope='Prepared locally with offline builds and exact Core/Data IL bridge; no AMD deployment, new-path execution or model/timing qualification yet')
    save(payload/'payload.json',spec)
    archive=BASE/'payload.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name in [*sorted(files),'payload.json']:tar.add(payload/name,arcname=name,recursive=False)
    save(BASE/'prepared.json',dict(passed=True,payload=pin(payload/'payload.json'),archive=pin(archive),files=len(files),
        bytes=sum(v['bytes'] for v in files.values()),nuget_uncompressed_bytes=unpacked,il_bridge=pin(BASE/'logs/local-il-bridge.json'),
        predecessors={str(p.relative_to(ROOT)):pin(p) for p in [ROWS/'prepared.json',ROWS/'local-preparation.json',PORTABLE/'qualification-closed.json',PRIOR/'closed.json']},scope=spec['scope']))
    print(json.dumps(read(BASE/'prepared.json')))


if __name__=='__main__':main()
