"""Extend the closed reuse consumer with read-only decoder ownership checks."""
import json,shutil
from prepare import ROOT,BASE,PRIOR,pin,write,run


def main():
    for path in [BASE/'local-closed.json',PRIOR/'closed.json']:
        receipt=json.loads(path.read_text());assert receipt['passed']
        for name,wanted in receipt['files'].items():assert pin(ROOT/name)==wanted,name
    assert json.loads((BASE/'local-final-verification.json').read_text())['passed']
    assert json.loads((PRIOR/'final-verification.json').read_text())['passed']
    folder=BASE/'consumer';folder.mkdir()
    text=(PRIOR/'consumer/Program.cs').read_text()
    def replace(a,b):
        nonlocal text
        assert text.count(a)==1,a;text=text.replace(a,b)
    replace('double setupSeconds=', 'var weightsBefore=Weights(whisper);\ndouble setupSeconds=')
    replace('File.WriteAllText(Path.Combine(output,"result.json")',
        'var weightsAfter=Weights(whisper);\nRequire(JsonSerializer.Serialize(weightsBefore)==JsonSerializer.Serialize(weightsAfter),"Decoder weights or storage changed");\nFile.WriteAllText(Path.Combine(output,"result.json")')
    replace('conformance,records,diagnostic=', 'conformance,records,weight_sharing=new{before=weightsBefore,after=weightsAfter},diagnostic=')
    inspector=(ROOT/'tests/whisper/weight-sharing/Inspect.cs').read_text()
    methods=inspector[inspector.index('static object Snapshot('):]
    methods=methods.replace('n.Inputs, n.Outputs','Inputs = n.Inputs.ToArray(), Outputs = n.Outputs.ToArray()')
    methods+='''
static object Weights(WhisperTranscriber transcriber)
{
    var flags=BindingFlags.Instance|BindingFlags.NonPublic;
    var first=(ComputationalGraph)typeof(WhisperTranscriber).GetField("firstDecoder",flags)!.GetValue(transcriber)!;
    var past=(ComputationalGraph)typeof(WhisperTranscriber).GetField("pastDecoder",flags)!.GetValue(transcriber)!;
    long logical=(long)typeof(WhisperTranscriber).GetProperty("SharedDecoderWeightBytes",flags)!.GetValue(transcriber)!;
    var firstArrays=new HashSet<Array>(first.Initializers.Values.Select(RootArray),ReferenceEqualityComparer.Instance);
    var shared=new Dictionary<Array,long>(ReferenceEqualityComparer.Instance);
    foreach(var value in past.Initializers.Values)
    {
        var array=RootArray(value);
        if(firstArrays.Contains(array))shared.TryAdd(array,Payload(value));
    }
    long sharedBytes=shared.Values.Sum();var physical=Physical(first,past);
    Require(logical==635187200,"Unexpected shared logical payload");
    Require(sharedBytes>=logical,"Actual shared storage missing");
    return new{logical_shared_bytes=logical,shared_arrays=shared.Count,shared_payload_bytes=sharedBytes,
        unique_arrays=physical.Count,unique_payload_bytes=physical.Bytes,first=Snapshot(first),past=Snapshot(past)};
}
'''
    replace('sealed record Case',methods+'\nsealed record Case')
    assert 'GC.Collect(' not in text and 'WaitForPendingFinalizers' not in text
    (folder/'Program.cs').write_text(text,encoding='utf-8')
    shutil.copyfile(PRIOR/'consumer/NpySupport.cs',folder/'NpySupport.cs')
    project=(PRIOR/'consumer/WhisperBufferReuse.csproj').read_text()
    (folder/'WhisperWeightSharing.csproj').write_text(project,encoding='utf-8')
    run(['dotnet','build',str(folder/'WhisperWeightSharing.csproj'),'--tl:off','--nologo','-v','minimal','-c','Release','-o',str(BASE/'bin')],'consumer-build.log')
    for p in (BASE/'product-bin').iterdir():
        target=BASE/'bin'/p.name
        if target.exists():assert pin(target)==pin(p)
        else:shutil.copyfile(p,target)
    paths=list((BASE/'bin').iterdir())+[folder/n for n in ['Program.cs','NpySupport.cs','WhisperWeightSharing.csproj']]+[BASE/'consumer-build.log',BASE/'built.json']
    write(BASE/'prepared.json',dict(prepared=True,vm_started=False,explicit_gc=False,local_closure=pin(BASE/'local-closed.json'),
        prior_closure=pin(PRIOR/'closed.json'),files={p.relative_to(BASE).as_posix():pin(p) for p in paths}))
    print(json.dumps(dict(prepared=True,receipt=pin(BASE/'prepared.json'))))


if __name__=='__main__':main()
