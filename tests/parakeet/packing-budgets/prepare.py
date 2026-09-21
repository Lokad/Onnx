"""Build Data-only budget prototypes against the exact qualified Core."""
import difflib
from common import *


def main():
    for p,expected in [(ADMISSION/'prepared.json','71ae4df6ca8b244263c476b8e6c747de6015eb6e48937f19a204f85a0c6e5d3c'),
        (QUALIFIED/'closed.json','0eabd1904ccaadefcc6beae594023206dfc57bff3c429a2f0ddb40b91dba94f1'),
        (NATIVE_BASELINE/'closed.json','57ded6ece06f486c31689ef14143237a0d188bcd76ced094a39932a45c406817')]:
        assert pin(p)['sha256']==expected;verify(read(p)['files'])
    assert pin(PRODUCT/'Lokad.Onnx.dll')['sha256']=='737e90134c8220a70d24caff6798bcdb430700295ea17b36479c2ae3d507fecf'
    BASE.mkdir();(BASE/'logs').mkdir();own=psutil.Process()
    state=dict(complete=False,code=None,supervisor=dict(pid=own.pid,birth=own.create_time()),runs=[])
    def command(label,args,cwd):return worker(state,BASE/'builds.json',label,args,cwd,[0],8,8,300,True,None)
    def build(label,project,cwd):
        command(label+'-restore',['dotnet','restore',project,*FLAGS,'--source',FEED,'--packages',BASE/'packages','-p:NuGetAudit=false','-p:FrozenProductDirectory='+str(PRODUCT)],cwd)
        command(label+'-build',['dotnet','build',project,'-c','Release',*FLAGS,'--no-restore','--disable-build-servers','-p:FrozenProductDirectory='+str(PRODUCT)],cwd)
    bridge=BASE/'bridge';bridge.mkdir();bridge_original=ROOT/'tests/whisper/memory-product-v2/CompareIlStable.cs'
    code=bridge_original.read_text(encoding='utf8')
    old='    if (differences.Length != 0) throw new InvalidDataException("Runtime IL differs: " + string.Join("; ", differences));'
    new='''    string[] expected = name == "Lokad.Onnx.Data.dll" ? ["Lokad.Onnx.ParakeetTranscriber::.ctor::Void .ctor(System.String)"] : [];
    if (!differences.SequenceEqual(expected)) throw new InvalidDataException("Unexpected methods: " + string.Join("; ", differences));'''
    assert code.count(old)==1;code=code.replace(old,new)
    old='        normalized_methods = oldMethods, equal = true });'
    assert code.count(old)==1;code=code.replace(old,'''        normalized_methods = oldMethods, unchanged_methods = oldMethods.Count - differences.Length,
        changed_methods = differences.ToDictionary(k => k, k => newMethods[k]), equal_except_budget = true });''')
    code=code.replace('All Core/Data method IL, resolved operands, locals, stack and exception clauses',
                      'Only Parakeet constructor budget may differ; every other method matches')
    (bridge/'Program.cs').write_text(code,encoding='utf8')
    (bridge/'Bridge.csproj').write_text('<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><ImplicitUsings>enable</ImplicitUsings><Nullable>enable</Nullable></PropertyGroup></Project>\n')
    variants={}
    try:
        build('bridge',bridge/'Bridge.csproj',bridge)
        for mib in (512,2032):
            label=str(mib);source=BASE/('data-source-'+label)
            shutil.copytree(DATA_SOURCE,source,ignore=shutil.ignore_patterns('bin','obj'))
            path=source/'ParakeetTranscriber.cs';before=path.read_text(encoding='utf8')
            old='encoder = Load(modelDirectory, "encoder-model.onnx", 256L * 1024 * 1024);'
            assert before.count(old)==1;after=before.replace(old,old.replace('256L',label+'L'));path.write_text(after,encoding='utf8')
            (BASE/('budget-'+label+'.patch')).write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),
                fromfile='a/src/Lokad.Onnx.Data/ParakeetTranscriber.cs',tofile='b/src/Lokad.Onnx.Data/ParakeetTranscriber.cs')),encoding='utf8')
            project=source/'Lokad.Onnx.Data.csproj';text=project.read_text(encoding='utf8')
            old='    <ProjectReference Include="..\\Lokad.Onnx\\Lokad.Onnx.csproj" />'
            assert text.count(old)==1;text=text.replace(old,'''    <Reference Include="Lokad.Onnx"><HintPath>$(FrozenProductDirectory)/Lokad.Onnx.dll</HintPath></Reference>
    <Reference Include="Google.Protobuf"><HintPath>$(FrozenProductDirectory)/Google.Protobuf.dll</HintPath></Reference>''')
            project.write_text(text,encoding='utf8');build('data-'+label,project,source)
            runtime=BASE/('runtime-'+label);shutil.copytree(PRODUCT,runtime)
            shutil.copy2(source/'bin/Release/net10.0/Lokad.Onnx.Data.dll',runtime/'Lokad.Onnx.Data.dll')
            assert pin(runtime/'Lokad.Onnx.dll')==pin(PRODUCT/'Lokad.Onnx.dll')
            command('instructions-'+label,['dotnet',bridge/'bin/Release/net10.0/Bridge.dll',PRODUCT,runtime,BASE/('instructions-'+label+'.json')],ROOT)
            instruction=read(BASE/('instructions-'+label+'.json'));assert instruction['passed']
            data=instruction['observations'][1];assert data['unchanged_methods']==690
            key=next(iter(data['changed_methods']));left=json.loads(data['normalized_methods'][key]);right=json.loads(data['changed_methods'][key])
            assert {k:v for k,v in left.items() if k!='instructions'}=={k:v for k,v in right.items() if k!='instructions'}
            changes=[(a,b) for a,b in zip(left['instructions'],right['instructions'],strict=True) if a!=b];assert len(changes)==1,changes
            a,b=changes[0];assert a['opcode']==b['opcode']=='ldc.i4' and a['offset']==b['offset']
            assert int.from_bytes(bytes.fromhex(a['operand']),'little')==256*1024**2
            assert int.from_bytes(bytes.fromhex(b['operand']),'little')==mib*1024**2
            consumer=BASE/('consumer-'+label);consumer.mkdir()
            original=ROOT/'artifacts/parakeet-packing-qualification-20260921/source'
            program=(original/'Program.cs').read_text(encoding='utf8');old=pin(PRODUCT/'Lokad.Onnx.Data.dll')['sha256'];new=pin(runtime/'Lokad.Onnx.Data.dll')['sha256']
            assert program.count(old)==1;program=program.replace(old,new).replace('fixed 256 MiB encoder cap','fixed '+label+' MiB encoder cap')
            (consumer/'Program.cs').write_text(program,encoding='utf8')
            for name in ('Profile.csproj','NpySupport.cs'):shutil.copy2(original/name,consumer/name)
            build('consumer-'+label,consumer/'Profile.csproj',consumer)
            for p in (consumer/'bin/Release/net10.0').iterdir():
                if p.is_file() and p.name.startswith('Profile.'):shutil.copy2(p,runtime/p.name)
            for p in (NATIVE/'replay').iterdir():
                if p.is_file() and p.name.startswith('TranscribeReplay.'):shutil.copy2(p,runtime/p.name)
            assert pin(runtime/'TranscribeReplay.dll')['sha256']=='335ca09d0e45e344068c484c92af9d0db43a6ae0accd1895d7ae7bb88b0afcf9'
            variants[label]=dict(core=pin(runtime/'Lokad.Onnx.dll'),data=pin(runtime/'Lokad.Onnx.Data.dll'),
                profile=pin(runtime/'Profile.dll'),native=pin(runtime/'TranscribeReplay.dll'),retained_bytes=mib*1024**2,weights=56 if mib==512 else 217)
        state['code']=0
    except BaseException:state.update(code=1,error=traceback.format_exc());raise
    finally:state['complete']=True;save(BASE/'builds.json',state)
    files={}
    for p in BASE.rglob('*'):
        if p.is_file() and not {'obj','packages'}.intersection(p.relative_to(BASE).parts):files[p.relative_to(ROOT).as_posix()]=pin(p)
    for p in TOOLS.iterdir():
        if p.is_file():files[p.relative_to(ROOT).as_posix()]=pin(p)
    for p in (ADMISSION/'prepared.json',QUALIFIED/'closed.json',NATIVE_BASELINE/'closed.json',MANIFEST,REFERENCE,bridge_original):files[p.relative_to(ROOT).as_posix()]=pin(p)
    spec=read(MANIFEST)
    for value in [*spec['models'].values(),spec['reference'],*[c['pcm'] for c in spec['cases']]]:
        assert pin(ROOT/value['path'])=={k:value[k] for k in ('bytes','sha256')};files[value['path']]=pin(ROOT/value['path'])
    for name,value in read(REFERENCE)['files'].items():
        p=REFERENCE.parent/name;assert pin(p)=={k:value[k] for k in ('bytes','sha256')};files[p.relative_to(ROOT).as_posix()]=pin(p)
    save(BASE/'prepared.json',dict(passed=True,files=files,variants=variants,
        scope='Only encoder cap differs from qualified candidate; higher-budget inference still pending'))
    print(json.dumps(dict(passed=True,variants=variants,prepared=pin(BASE/'prepared.json'))))


if __name__=='__main__':main()
