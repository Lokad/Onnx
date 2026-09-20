"""Build a read-only normal-runtime memory observer around the original public consumer."""
import json,shutil
from prepare import ROOT,BASE,pin,run


def main():
    built=json.loads((BASE/'built.json').read_text());assert built['tests_passed']
    for name,wanted in built['files'].items():assert pin(BASE/name)==wanted,name
    folder=BASE/'consumer';assert not folder.exists();folder.mkdir()
    original=ROOT/'tests/audio/whisper-comparison/Program.cs';text=original.read_text()
    def replace(a,b):
        nonlocal text
        assert text.count(a)==1,a;text=text.replace(a,b)
    replace('    var gcBefore=', '    var memoryBefore=Memory();\n    var gcBefore=')
    replace('    var w=(WhisperTranscription)actual;', '    var memoryAfter=Memory();var pools=Pools(whisper);\n    var w=(WhisperTranscription)actual;')
    replace('allocated_bytes=allocatedAfter-allocated};','allocated_bytes=allocatedAfter-allocated,memory_before=memoryBefore,memory_after=memoryAfter,pools};')
    replace('conformance,records,setup_seconds=', 'conformance,records,diagnostic="bounded-context-reuse-no-forced-gc",setup_seconds=')
    memory=(ROOT/'tests/whisper/memory-collection/Program.cs').read_text().split('static object Memory()')[1].split('sealed record Case')[0]
    methods='''static object Pools(WhisperTranscriber transcriber)
{
    var result=new Dictionary<string,object>();
    foreach(var (name,budget) in new[]{("encodingExecution",512L*1024*1024),("firstExecution",128L*1024*1024),("pastExecution",128L*1024*1024)})
    {
        var context=(GraphExecution)typeof(WhisperTranscriber).GetField(name,BindingFlags.Instance|BindingFlags.NonPublic)!.GetValue(transcriber)!;
        var cache=typeof(ComputationalGraph).GetField("ReleasedBuffers",BindingFlags.Instance|BindingFlags.NonPublic)!.GetValue(context)!;
        long bytes=(long)cache.GetType().GetProperty("Bytes",BindingFlags.Instance|BindingFlags.NonPublic)!.GetValue(cache)!;
        int count=(int)cache.GetType().GetProperty("Count",BindingFlags.Instance|BindingFlags.NonPublic)!.GetValue(cache)!;
        Require(bytes>=0 && bytes<=budget && count>=0 && count<=256,"Cache bound: "+name);
        result[name]=new{allocated_new_bytes=context.LastPoolAllocatedNewBytes,reused_bytes=context.LastPoolReusedBytes,cache_bytes=bytes,cache_count=count,cache_budget=budget};
    }
    return result;
}
static object Memory()'''+memory
    replace('sealed record Case',methods+'sealed record Case')
    assert 'GC.Collect(' not in text and 'WaitForPendingFinalizers' not in text
    (folder/'Program.cs').write_text(text,encoding='utf-8')
    shutil.copyfile(ROOT/'tests/Shared/NpySupport.cs',folder/'NpySupport.cs')
    project='''<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><TargetFramework>net10.0</TargetFramework><OutputType>Exe</OutputType><ImplicitUsings>enable</ImplicitUsings><Nullable>enable</Nullable><AllowUnsafeBlocks>true</AllowUnsafeBlocks></PropertyGroup><ItemGroup><Reference Include="Lokad.Onnx"><HintPath>../product-bin/Lokad.Onnx.dll</HintPath></Reference><Reference Include="Lokad.Onnx.Data"><HintPath>../product-bin/Lokad.Onnx.Data.dll</HintPath></Reference></ItemGroup></Project>'''
    (folder/'WhisperBufferReuse.csproj').write_text(project,encoding='utf-8')
    run(['dotnet','build',str(folder/'WhisperBufferReuse.csproj'),'--tl:off','--nologo','-v','minimal','-c','Release','-o',str(BASE/'bin')],BASE/'consumer-build.log')
    for p in (BASE/'product-bin').iterdir():
        target=BASE/'bin'/p.name
        if target.exists():assert pin(target)==pin(p)
        else:shutil.copyfile(p,target)
    paths=list((BASE/'bin').iterdir())+[folder/n for n in ['Program.cs','NpySupport.cs','WhisperBufferReuse.csproj']]+[BASE/'consumer-build.log',BASE/'built.json']
    value=dict(prepared=True,vm_started=False,built=pin(BASE/'built.json'),original_consumer=pin(original),conformance_calls=20,endurance_calls=80,explicit_gc=False,files={p.relative_to(BASE).as_posix():pin(p) for p in paths})
    with (BASE/'prepared.json').open('x') as f:json.dump(value,f,indent=2)
    print(json.dumps(dict(prepared=True,receipt=pin(BASE/'prepared.json'))))


if __name__=='__main__':main()
