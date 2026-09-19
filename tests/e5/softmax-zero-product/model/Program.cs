using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;
using Microsoft.ML.OnnxRuntime;

static string Digest(string path) => Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));
static long Affinity() { if (OperatingSystem.IsLinux() || OperatingSystem.IsWindows()) return Process.GetCurrentProcess().ProcessorAffinity.ToInt64(); throw new PlatformNotSupportedException(); }
static void Require(bool value, string message) { if (!value) throw new InvalidDataException(message); }
static void NoNativeOrt() => Require(!Process.GetCurrentProcess().Modules.Cast<ProcessModule>().Any(m =>
    Path.GetFileName(m.FileName).StartsWith("libonnxruntime", StringComparison.OrdinalIgnoreCase) ||
    Path.GetFileName(m.FileName).Equals("onnxruntime.dll", StringComparison.OrdinalIgnoreCase)), "Native ORT loaded");
string mode = args[4], role=args[5];
Require(mode=="memory" && (role is "controlA" or "controlB" or "candidate" or "codegen"), "Unknown product role");
Require(Digest(typeof(ComputationalGraph).Assembly.Location)=="187de61ad8f034b9b7ad2fb3490358443fa84334204720e81bc3546a31f3c8d4", "Frozen product differs");
Require((Environment.GetEnvironmentVariable("LOKAD_ONNX_SOFTMAX_ZERO_BLOCKS")=="1")== (role is "candidate" or "codegen"), "Product switch differs");
Require(mode is "default" or "memory" or "ort", "Unknown public options");
string root = Path.GetFullPath(args[0]), name = args[1], fixturePath = Path.GetFullPath(args[2]), output = Path.GetFullPath(args[3]);
Require(!File.Exists(output), "Existing output");
NoNativeOrt();
var assembly = Assembly.Load("Lokad.Onnx.Campaign");
var isolated = assembly.GetType("IsolatedE5", true) ?? throw new InvalidDataException("Missing producer type");
var method = isolated.GetMethod("Inputs", BindingFlags.NonPublic | BindingFlags.Static) ?? throw new InvalidDataException("Missing input method");
var inputs = method.Invoke(null, new object[] {name, Path.Combine(root,"models/multilingual-e5-small/sentencepiece.bpe.model")}) as Dictionary<string, ITensor>
    ?? throw new InvalidDataException("Missing inputs");
var hashMethod = assembly.GetType("Lokad.Onnx.Bench.CampaignEvidence", true)?.GetMethod("HashInputs", BindingFlags.NonPublic | BindingFlags.Static)
    ?? throw new InvalidDataException("Missing input identity");
string InputHash() => hashMethod.Invoke(null, new object[] {inputs}) as string ?? throw new InvalidDataException("Missing hash");
var inputBits = inputs.ToDictionary(p => p.Key, p => ((Tensor<long>)p.Value).ToArray());
string inputHash = InputHash();
string model = Path.Combine(root,"models/multilingual-e5-small/model.onnx");
string modelHash = Digest(model), tokenizerHash = Digest(Path.Combine(root,"models/multilingual-e5-small/sentencepiece.bpe.model"));
Require(modelHash == "ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665", "Unexpected model");
using var doc = JsonDocument.Parse(File.ReadAllText(fixturePath));
var fixture = doc.RootElement;
Require(fixture.GetProperty("case").GetString() == name && fixture.GetProperty("model_sha256").GetString() == modelHash &&
    fixture.GetProperty("tokenizer_sha256").GetString() == tokenizerHash && fixture.GetProperty("input_sha256").GetString() == inputHash, "Fixture identity");
var expected = fixture.GetProperty("outputs").EnumerateArray().Select(value => {
    string file = Path.Combine(Path.GetDirectoryName(fixturePath) ?? throw new InvalidDataException(), value.GetProperty("file").GetString() ?? throw new InvalidDataException());
    Require(Digest(file) == value.GetProperty("sha256").GetString(), "Oracle output changed");
    return (Name: value.GetProperty("name").GetString() ?? throw new InvalidDataException(), Dims: value.GetProperty("dims").EnumerateArray().Select(v=>v.GetInt32()).ToArray(),
        Data: MemoryMarshal.Cast<byte,float>(File.ReadAllBytes(file)).ToArray());
}).ToArray();
long loadStart=Stopwatch.GetTimestamp();
var opts = mode == "memory" ? ExecutionOptions.Memory : ExecutionOptions.Default;
Require(Environment.ProcessorCount == 1 && opts.Tensor.MaxDegreeOfParallelism == 1, "Confinement before CLR startup required");
ComputationalGraph? graph = null;
InferenceSession? session = null;
SessionOptions? sessionOptions = null;
RunOptions? runOptions = null;
var nativeInputs = new Dictionary<string, OrtValue>(StringComparer.Ordinal);
IDisposableReadOnlyCollection<OrtValue>? nativeResult = null;
string[] outputNames;
if (mode == "ort")
{
    sessionOptions = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
        IntraOpNumThreads = 1, InterOpNumThreads = 1, ExecutionMode = ExecutionMode.ORT_SEQUENTIAL };
    sessionOptions.AddSessionConfigEntry("session.intra_op.allow_spinning", "0");
    sessionOptions.AddSessionConfigEntry("session.inter_op.allow_spinning", "0");
    session = new InferenceSession(model, sessionOptions);
    runOptions = new RunOptions();
    foreach (var p in inputs) nativeInputs.Add(p.Key, OrtValue.CreateTensorValueFromMemory(((Tensor<long>)p.Value).ToArray(), p.Value.Dims.Select(d => (long)d).ToArray()));
    outputNames = session.OutputMetadata.Keys.Order(StringComparer.Ordinal).ToArray();
}
else
{
    graph = OnnxImport.Load(model) ?? throw new InvalidDataException("Import failed");
    outputNames = graph.OutputDescs.Select(t => t.Name).Order(StringComparer.Ordinal).ToArray();
}
long loadTicks=Stopwatch.GetTimestamp()-loadStart;
Require(outputNames.SequenceEqual(expected.Select(v => v.Name)), "Output set/order mismatch");
void Reset() { graph?.Reset(); nativeResult?.Dispose(); nativeResult = null; }
void Execute()
{
    if (graph != null) Require(graph.Execute(inputs, true, ExecutionProvider.CPU, opts), graph.LastErrorMessage ?? "Execute failed");
    else nativeResult = (session ?? throw new InvalidDataException()).Run(runOptions ?? throw new InvalidDataException(), nativeInputs, outputNames);
}
(int[] Dims, float[] Data) Output(int k)
{
    if (graph != null)
    {
        var tensor = (Tensor<float>)graph.Outputs[outputNames[k]];
        return (tensor.Dimensions.ToArray(), tensor.ToArray());
    }
    var value = (nativeResult ?? throw new InvalidDataException()).ElementAt(k);
    var shape = value.GetTensorTypeAndShape();
    Require(shape.ElementDataType == Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.Float, "Native output dtype");
    return (shape.Shape.Select(d => checked((int)d)).ToArray(), value.GetTensorDataAsSpan<float>().ToArray());
}
object? NativeIdentity()
{
    var modules=Process.GetCurrentProcess().Modules.Cast<ProcessModule>().Where(m =>
        Path.GetFileName(m.FileName).StartsWith("libonnxruntime",StringComparison.OrdinalIgnoreCase) ||
        Path.GetFileName(m.FileName).Equals("onnxruntime.dll",StringComparison.OrdinalIgnoreCase)).ToArray();
    if(mode!="ort") { Require(modules.Length==0,"Native ORT loaded in managed worker"); return null; }
    Require(modules.Length==1,"Expected one native module");
    var path=modules[0].FileName;var hash=Digest(path);var version=OrtEnv.Instance().GetVersionString();
    Require(version==fixture.GetProperty("oracle_version").GetString(),"Native version differs");
    if(OperatingSystem.IsLinux())Require(hash==fixture.GetProperty("native").GetProperty("sha256").GetString(),"Native bytes differ");
    return new {path,sha256=hash,version};
}
var nativeBefore=NativeIdentity();
double Validate(string stage)
{
    Reset(); Execute();
    double error=0;
    for (int k=0;k<expected.Length;k++)
    {
        var reference=expected[k]; var actualOutput=Output(k); var actual=actualOutput.Data;
        Require(reference.Dims.SequenceEqual(actualOutput.Dims) && reference.Data.Length==actual.Length, "Output geometry");
        for(int i=0;i<actual.Length;i++)
        {
            Require(float.IsFinite(actual[i]) && float.IsFinite(reference.Data[i]), "Nonfinite output");
            error=Math.Max(error,Math.Abs((double)actual[i]-reference.Data[i])/Math.Max(1,Math.Abs((double)reference.Data[i])));
        }
        string file=output+"."+stage+"-"+k+".f32";
        File.WriteAllBytes(file,MemoryMarshal.AsBytes(actual.AsSpan()).ToArray());
    }
    Require(error<=1e-4,"Native agreement");
    return error;
}
object Resources() { using var p=Process.GetCurrentProcess(); var m=GC.GetGCMemoryInfo(); return new {rss=p.WorkingSet64, peak=p.PeakWorkingSet64, managed=GC.GetTotalMemory(false), heap=m.HeapSizeBytes, committed=m.TotalCommittedBytes, fragmented=m.FragmentedBytes, gc=new[]{GC.CollectionCount(0),GC.CollectionCount(1),GC.CollectionCount(2)}}; }
long firstStart=Stopwatch.GetTimestamp(); Execute(); long firstTicks=Stopwatch.GetTimestamp()-firstStart;
var afterFirst=Resources();
// Hold the first returned tensor through every subsequent request.
var held=graph?.Outputs.ToDictionary(p=>p.Key,p=>p.Value as Tensor<float> ?? throw new InvalidDataException("Missing first output"));
var heldNative=nativeResult; nativeResult=null;
var heldBits=held != null ? outputNames.Select(n=>held[n].ToArray()).ToArray() :
    (heldNative ?? throw new InvalidDataException()).Select(v=>v.GetTensorDataAsSpan<float>().ToArray()).ToArray();
double beforeError=Validate("before");
var conditioning=new List<long>(); double conditioningSeconds=0;
var beforeConditioning=Resources();
while(conditioningSeconds<30)
{
    Reset(); long start=Stopwatch.GetTimestamp(); Execute(); long elapsed=Stopwatch.GetTimestamp()-start;
    conditioning.Add(elapsed); conditioningSeconds+=(double)elapsed/Stopwatch.Frequency;
}
var afterConditioning=Resources();
object Block(bool includeReset)
{
    var result=new long[33];var before=Resources();long allocated=GC.GetTotalAllocatedBytes(true);
    for(int i=0;i<result.Length;i++)
    {
        if(!includeReset)Reset();
        long start=Stopwatch.GetTimestamp();if(includeReset)Reset();Execute();result[i]=Stopwatch.GetTimestamp()-start;
    }
    long bytes=GC.GetTotalAllocatedBytes(true)-allocated;var after=Resources();
    return new {include_reset=includeReset,ticks=result,allocated_bytes=bytes,before,after};
}
var executeBlock=Block(false);var requestBlock=Block(true);
double afterError=Validate("after");
Require(InputHash()==inputHash,"Input digest changed");
foreach(var p in inputs)Require(inputBits[p.Key].SequenceEqual(((Tensor<long>)p.Value).ToArray()),"Input bits changed");
for(int k=0;k<heldBits.Length;k++)
{
    var current=held != null ? held[outputNames[k]].ToArray() : (heldNative ?? throw new InvalidDataException()).ElementAt(k).GetTensorDataAsSpan<float>().ToArray();
    Require(MemoryMarshal.AsBytes(current.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(heldBits[k].AsSpan())),"Held output bits changed");
}
foreach(var p in nativeInputs)Require(p.Value.GetTensorDataAsSpan<long>().SequenceEqual(inputBits[p.Key]),"Actual native input changed");
var nativeAfter=NativeIdentity();
Require(JsonSerializer.Serialize(nativeBefore)==JsonSerializer.Serialize(nativeAfter),"Native identity changed");
File.WriteAllText(output,JsonSerializer.Serialize(new{diagnostic_only=true,schema=3,protocol="zero-block-product-descriptive-v1",name,mode,role,optimization=mode=="ort" ? "ORT_ENABLE_ALL" : opts.Optimization.ToString(),native=nativeAfter,parallelism=opts.Tensor.MaxDegreeOfParallelism,
    core_sha256=Digest(typeof(ComputationalGraph).Assembly.Location),probe_sha256=Digest(Assembly.GetExecutingAssembly().Location),runner_sha256=Digest(assembly.Location),fixture_sha256=Digest(fixturePath),
    model_sha256=modelHash,tokenizer_sha256=tokenizerHash,input_sha256=inputHash,inputs=inputBits,
    frequency=Stopwatch.Frequency,affinity=Affinity(),runtime=RuntimeInformation.FrameworkDescription,
    avx2=Avx2.IsSupported,avx512=Avx512F.IsSupported,vector_width=System.Numerics.Vector<float>.Count,load_ticks=loadTicks,first_execute_ticks=firstTicks,after_first=afterFirst,
    conditioning,conditioning_seconds=conditioningSeconds,before_conditioning=beforeConditioning,after_conditioning=afterConditioning,
    before_error=beforeError,after_error=afterError,execute=executeBlock,request=requestBlock,held_outputs_unchanged=true,inputs_unchanged=true,final_resources=Resources(),
    settings=Environment.GetEnvironmentVariables().Keys.Cast<string>().Where(k=>k.StartsWith("LOKAD_",StringComparison.Ordinal)||k.StartsWith("DOTNET_",StringComparison.Ordinal)||k.StartsWith("COMPlus_",StringComparison.Ordinal)).ToDictionary(k=>k,Environment.GetEnvironmentVariable)
},new JsonSerializerOptions{WriteIndented=true}));
Reset(); heldNative?.Dispose(); foreach(var value in nativeInputs.Values)value.Dispose();
runOptions?.Dispose(); session?.Dispose(); sessionOptions?.Dispose();
Console.WriteLine("Complete public/native configuration "+name+" "+mode);
