"""Build a separate public contract consumer using the already tested private product."""
from pathlib import Path
import hashlib,json,shutil,subprocess

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/whisper-memory-contracts-20260920'
PRODUCT=ROOT/'artifacts/whisper-weight-sharing-20260920'
PRIOR=ROOT/'artifacts/whisper-recording-v2-20260919'


def pin(p):
    p=Path(p)
    with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)


def main():
    assert pin(PRIOR/'receipt.json')['sha256']=='623d1ad90efa120d0910f1f9fad92deaf3d4d9ad8410b74e8c10f16742d66275'
    prior=read(PRIOR/'receipt.json');local=read(PRODUCT/'local-closed.json');assert local['passed']
    for name,wanted in local['files'].items():assert pin(ROOT/name)==wanted,name
    assert read(PRODUCT/'local-final-verification.json')['passed']
    for name in ['frozen.json','inputs/inputs.json','native-corrected/manifest.json','managed/result.json','audit.json','native-audit.json']:
        assert pin(PRIOR/name)['sha256']==prior['files'][name],name
    BASE.mkdir();source=BASE/'source';source.mkdir();product=BASE/'product-bin';product.mkdir()
    for path in (PRODUCT/'product-bin').iterdir():shutil.copyfile(path,product/path.name)
    original=PRIOR/'source/tests/whisper/recording/Program.cs'
    assert pin(original)['sha256']==read(PRIOR/'frozen.json')['source']['tests/whisper/recording/Program.cs']
    text=original.read_text(encoding='utf-8')
    def replace(a,b):
        nonlocal text
        assert text.count(a)==1,a;text=text.replace(a,b)
    replace('string models=','using var running=Process.GetCurrentProcess();\nRequire(OperatingSystem.IsWindows() || OperatingSystem.IsLinux(),"Affinity platform");\nRequire(running.ProcessorAffinity.ToInt64()==4,"CPU2 before CLR required");\nstring models=')
    replace('var options=WhisperRecordingOptions.ForLanguage("en");','var weightsBefore=Weights(model);\nvar options=WhisperRecordingOptions.ForLanguage("en");')
    insert='''
var speechCases=old.RootElement.GetProperty("cases").EnumerateArray().Take(2).ToArray();
var speechPcm=speechCases.Select(c=>
{
    string path=Path.Combine(Path.GetDirectoryName(shortReference)!,c.GetProperty("pcm").GetString()!);
    Require(Sha(path)==c.GetProperty("pcm_sha256").GetString(),"Concurrent speech input hash");
    return NpySupport.ReadFloat32(path).Values;
}).ToArray();
var speechOriginal=speechPcm.Select(p=>MemoryMarshal.AsBytes(p.AsSpan()).ToArray()).ToArray();
using var ready=new CountdownEvent(2);using var begin=new ManualResetEventSlim();
var speechTasks=Enumerable.Range(0,2).Select(index=>Task.Factory.StartNew(()=>
{
    ready.Signal();begin.Wait();long start=Stopwatch.GetTimestamp();
    var result=model.Transcribe(speechPcm[index],16000,WhisperTranscriptionOptions.ForLanguage("en"),CancellationToken.None);
    long end=Stopwatch.GetTimestamp();return new SpeechCall(index,start,end,speechCases[index].GetProperty("pcm_sha256").GetString()!,result);
},CancellationToken.None,TaskCreationOptions.LongRunning,TaskScheduler.Default)).ToArray();
Require(ready.Wait(TimeSpan.FromSeconds(30)),"Concurrent speech threads failed to start");begin.Set();
var concurrentSpeech=await Task.WhenAll(speechTasks);
Require(concurrentSpeech.Max(r=>r.Start)<concurrentSpeech.Min(r=>r.End),"Speech request lifetimes did not overlap");
var speechSnapshots=concurrentSpeech.Select(r=>JsonSerializer.Serialize(r.Result,json)).ToArray();
for(int i=0;i<2;i++)
{
    var actual=concurrentSpeech[i].Result;var expected=speechCases[i];
    Require(actual.Text==expected.GetProperty("text").GetString()
        && actual.TokenIds.SequenceEqual(expected.GetProperty("tokens").EnumerateArray().Select(t=>t.GetInt32()))
        && actual.StopReason.ToString()==expected.GetProperty("stop_reason").GetString()
        && actual.SkippedAsNoSpeech==expected.GetProperty("skipped_as_no_speech").GetBoolean(),"Concurrent speech reference differs");
    Require(MemoryMarshal.AsBytes(speechPcm[i].AsSpan()).SequenceEqual(speechOriginal[i]),"Concurrent speech PCM changed");
}
Require(speechSnapshots[0]==JsonSerializer.Serialize(shortResult,json),"Concurrent repeated short result differs");
Reject<ArgumentOutOfRangeException>(()=>model.Transcribe(shortPcm,8000,WhisperTranscriptionOptions.ForLanguage("en"),CancellationToken.None));
Reject<ArgumentException>(()=>model.Transcribe(new[]{float.NaN},16000,WhisperTranscriptionOptions.ForLanguage("en"),CancellationToken.None));
Reject<ArgumentNullException>(()=>model.Transcribe(shortPcm,16000,null!,CancellationToken.None));
Reject<ArgumentOutOfRangeException>(()=>model.Transcribe(new float[480001],16000,WhisperTranscriptionOptions.ForLanguage("en"),CancellationToken.None));
using(var canceledAfterWork=new CancellationTokenSource())
{
    canceledAfterWork.Cancel();Reject<OperationCanceledException>(()=>model.Transcribe(shortPcm,16000,WhisperTranscriptionOptions.ForLanguage("en"),canceledAfterWork.Token));
}
using(var interruptedAfterWork=new CancellationTokenSource())
{
    interruptedAfterWork.CancelAfter(50);Reject<OperationCanceledException>(()=>model.Transcribe(shortPcm,16000,WhisperTranscriptionOptions.ForLanguage("en"),interruptedAfterWork.Token));
}
var shortRecovery=model.Transcribe(shortPcm,16000,WhisperTranscriptionOptions.ForLanguage("en"),CancellationToken.None);
Require(JsonSerializer.Serialize(shortRecovery,json)==JsonSerializer.Serialize(shortResult,json),"Canceled short request changed recovery");
for(int i=0;i<2;i++)Require(speechSnapshots[i]==JsonSerializer.Serialize(concurrentSpeech[i].Result,json)
    && MemoryMarshal.AsBytes(speechPcm[i].AsSpan()).SequenceEqual(speechOriginal[i]),"Held concurrent result or PCM changed");
Require(refusals==16,"Refusal coverage differs");
var weightsAfter=Weights(model);
Require(JsonSerializer.Serialize(weightsBefore)==JsonSerializer.Serialize(weightsAfter),"Decoder values or shared storage changed");
File.WriteAllText(Path.Combine(destination,"weights.json"),JsonSerializer.Serialize(new{before=weightsBefore,after=weightsAfter},new JsonSerializerOptions{WriteIndented=true}));
'''
    marker='Require(MemoryMarshal.AsBytes(shortPcm.AsSpan()).SequenceEqual(shortOriginal),"Short API input changed");'
    replace(marker,insert+'\n'+marker)
    replace('short_regression=shortResult,', 'concurrent_speech=concurrentSpeech,short_recovery=shortRecovery,frequency=Stopwatch.Frequency,weights_sha256=Sha(Path.Combine(destination,"weights.json")),\n    affinity=running.ProcessorAffinity.ToInt64(),processor_count=Environment.ProcessorCount,short_regression=shortResult,')
    methods=(PRODUCT/'consumer/Program.cs').read_text(encoding='utf-8').split('static object Snapshot(',1)[1].split('sealed record Case',1)[0]
    text+='\n'+ 'static object Snapshot('+methods+'\nsealed record SpeechCall(int Index,long Start,long End,string PcmSha256,WhisperTranscription Result);\n'
    assert 'GC.Collect(' not in text
    (source/'Program.cs').write_text(text,encoding='utf-8')
    shutil.copyfile(ROOT/'tests/Shared/NpySupport.cs',source/'NpySupport.cs')
    shutil.copyfile(PRIOR/'source/tests/whisper/transcription-assets.json',source/'assets.json')
    project='''<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><Nullable>enable</Nullable><ImplicitUsings>enable</ImplicitUsings></PropertyGroup><ItemGroup><Reference Include="Lokad.Onnx"><HintPath>../product-bin/Lokad.Onnx.dll</HintPath></Reference><Reference Include="Lokad.Onnx.Data"><HintPath>../product-bin/Lokad.Onnx.Data.dll</HintPath></Reference><EmbeddedResource Include="assets.json" LogicalName="assets.json" /></ItemGroup></Project>'''
    (source/'WhisperMemoryContracts.csproj').write_text(project,encoding='utf-8')
    with (BASE/'build.log').open('x') as log:r=subprocess.run(['dotnet','build',str(source/'WhisperMemoryContracts.csproj'),'--tl:off','--nologo','-v','minimal','-c','Release','-o',str(BASE/'bin')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    assert r.returncode==0,'Consumer build'
    for path in product.iterdir():
        target=BASE/'bin'/path.name
        if target.exists():assert pin(target)==pin(path)
        else:shutil.copyfile(path,target)
    inputs={}
    def add(path):inputs[path.relative_to(ROOT).as_posix()]=pin(path)
    for name in ['receipt.json','frozen.json','inputs/inputs.json','native-corrected/manifest.json','managed/result.json','audit.json','native-audit.json']:add(PRIOR/name)
    for row in read(PRIOR/'inputs/inputs.json')['cases']:
        for key in ['pcm','wave']:
            path=PRIOR/'inputs'/row[key];assert pin(path)['sha256']==row[key+'_sha256'];add(path)
    short=ROOT/'artifacts/asr-labeled-20260919/native-whisper/manifest.json'
    assert pin(short)['sha256']==read(PRIOR/'managed/result.json')['short_manifest_sha256'];add(short)
    for row in read(short)['cases'][:2]:
        path=short.parent/row['pcm'];assert pin(path)['sha256']==row['pcm_sha256'];add(path)
    for name,wanted in read(source/'assets.json')['files'].items():
        path=ROOT/'models/whisper-large-v3-turbo'/name;assert pin(path)==wanted,name;add(path)
    shutil.copyfile(ROOT/'.agent/m4-whisper-memory-contracts-20260920.md',BASE/'prospective-plan.md')
    paths=[*sorted((BASE/'bin').iterdir()),*sorted(source.glob('*.*')),BASE/'build.log',BASE/'prospective-plan.md']
    write(BASE/'prepared.json',dict(prepared=True,product_closure=pin(PRODUCT/'local-closed.json'),original_program=pin(original),
        files={p.relative_to(BASE).as_posix():pin(p) for p in paths},inputs=inputs,completed_requests=13,refusals=16))
    print(json.dumps(dict(prepared=True,receipt=pin(BASE/'prepared.json'))))


if __name__=='__main__':main()
