using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

internal static class ModelReplay
{
    static void Require(bool value, string message) { if (!value) throw new InvalidDataException(message); }
    static string Hash(float[] values) => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(values.AsSpan())));
    static string Sha(string path) { using var f = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(f)); }
    static DenseTensor<float> Load(string directory, JsonElement item)
    {
        string path = Path.Combine(directory, item.GetProperty("file").GetString()!);
        Require(Sha(path) == item.GetProperty("sha256").GetString() && new FileInfo(path).Length == item.GetProperty("bytes").GetInt64(), "Tensor identity");
        var values = MemoryMarshal.Cast<byte,float>(File.ReadAllBytes(path)).ToArray();
        return new DenseTensor<float>(values, item.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32()).ToArray());
    }
    static int Main(string[] args)
    {
        if (!OperatingSystem.IsWindows() && !OperatingSystem.IsLinux()) throw new PlatformNotSupportedException();
        Require(args.Length == 5, "fixture-root output expected-core selected-or-candidate 256-or-512-or-scalar");
        Require(Environment.ProcessorCount == 1 && Process.GetCurrentProcess().ProcessorAffinity.ToInt64() == 4, "CPU2 before CLR startup");
        Require(Sha(typeof(CPUExecutionProvider).Assembly.Location) == args[2] && !File.Exists(args[1]), "Core / destination");
        Require(args[3] is "selected" or "candidate" && args[4] is "256" or "512" or "scalar", "Mode");
        var flags = Environment.GetEnvironmentVariables().Keys.Cast<string>().Where(k => k.StartsWith("DOTNET_",StringComparison.OrdinalIgnoreCase)
            || k.StartsWith("COMPlus_",StringComparison.OrdinalIgnoreCase) || k.StartsWith("LOKAD_",StringComparison.OrdinalIgnoreCase)).Order().ToArray();
        Require(flags.All(k => (args[4] == "256" && k == "DOTNET_EnableAVX512") || (args[4] == "scalar" && k == "DOTNET_EnableHWIntrinsic")), "Flag allowlist");
        Require(flags.All(k => Environment.GetEnvironmentVariable(k) == "0"), "Disable flag values");
        Require(Avx512F.IsSupported == (args[4] == "512") && Vector.IsHardwareAccelerated == (args[4] != "scalar"), "Actual hardware route");
        string folder = Path.Combine(args[0], "output"), nativeFolder = Path.Combine(args[0], "native");
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(folder,"result.json")));
        using var reference = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(nativeFolder,"result.json")));
        Require(document.RootElement.GetProperty("passed").GetBoolean() && reference.RootElement.GetProperty("passed").GetBoolean(), "Qualified fixtures");
        var calls = document.RootElement.GetProperty("calls").EnumerateArray().ToArray();
        var natives = reference.RootElement.GetProperty("reports").EnumerateArray().ToArray();
        Require(calls.Length == 12 && natives.Length == 36, "Full census");
        var observations = new List<object>(); var held = new List<(Tensor<float> Tensor, string Hash)>();
        void CheckHeld() { foreach (var item in held) Require(Hash(item.Tensor.ToArray()) == item.Hash,"Held output changed"); }
        long values = 0; double maximum = 0;
        for (int ordinal = 0; ordinal < calls.Length; ordinal++)
        {
            var call = calls[ordinal]; string name = call.GetProperty("name").GetString()!; int index = call.GetProperty("index").GetInt32();
            Require(index == ordinal % 4 && call.GetProperty("opset").GetInt32() == 17, "Original order/opset");
            var names = call.GetProperty("input_names").EnumerateArray().Select(v => v.GetString()!).ToArray();
            var outNames = call.GetProperty("output_names").EnumerateArray().Select(v => v.GetString()!).ToArray();
            var items = call.GetProperty("inputs").EnumerateArray().ToArray(); var outputs = call.GetProperty("outputs").EnumerateArray().ToArray();
            Require(names.Length == 7 && names[4] == "" && outNames.Length == 3, "Original optional slots");
            var tensors = items.Select(v => v.ValueKind == JsonValueKind.Null ? null : Load(folder,v)).ToArray();
            var before = tensors.Select(t => t is null ? null : Hash(t.ToArray())).ToArray();
            var attributes = call.GetProperty("attributes"); Require(attributes.EnumerateObject().Count() == 2, "Original attributes");
            Require(attributes.GetProperty("direction").GetString() == "bidirectional" && attributes.GetProperty("hidden_size").GetInt32() == 128, "Model dimensions");
            var graph = new ComputationalGraph(32L * 1024 * 1024); graph.Opset[""] = 17; graph.Inputs.Add(names[0],null);
            for (int i = 1; i < names.Length; i++) if (tensors[i] is not null) graph.Initializers.Add(names[i],tensors[i]!);
            foreach (string output in outNames) graph.Outputs.Add(output,null);
            graph.Nodes.Add(new Node { ID=ordinal+1,Name=call.GetProperty("node").GetString()!,Op=OpType.LSTM,OpTypeName="LSTM",Domain="",OpsetVersion=17,
                Inputs=names,Outputs=outNames,Attributes=new() { ["direction"]="bidirectional",["hidden_size"]=128 } });
            graph.Prepare(); var scratch = new ScratchAccountant();
            var options = ExecutionOptions.Memory with { Tensor = TensorExecutionOptions.Auto with { ScratchReporter=scratch } };
            var execution = graph.CreateExecution(options);
            for (int repeat = 0; repeat < 2; repeat++)
            {
                CheckHeld(); execution.Reset(); long previous = scratch.TotalScratchBytes;
                Require(execution.Execute(new Dictionary<string,ITensor> { [names[0]]=tensors[0]! },true,ExecutionProvider.CPU,options),execution.LastErrorMessage ?? "LSTM graph");
                long expectedScratch = Vector.IsHardwareAccelerated ? (tensors[1]!.Length+tensors[2]!.Length)*4+(args[3]=="candidate"?8192:0) : 0;
                Require(scratch.TotalScratchBytes-previous == expectedScratch,"Exact weight-panel and input-block scratch");
                for (int slot = 0; slot < 3; slot++)
                {
                    var output = (Tensor<float>)execution.Outputs[outNames[slot]]!; var actual = output.ToArray();
                    Require(Hash(actual) == outputs[slot].GetProperty("sha256").GetString(),"Selected full output bits");
                    var native = natives[ordinal*3+slot]; Require(native.GetProperty("case").GetString() == name && native.GetProperty("index").GetInt32() == index && native.GetProperty("slot").GetInt32() == slot,"Native reference order");
                    var expected = Load(nativeFolder,native.GetProperty("reference")); Require(output.Dimensions.SequenceEqual(expected.Dimensions),"Native output shape");
                    double error = 0; var refValues = expected.ToArray();
                    for (int i = 0; i < actual.Length; i++) { Require(float.IsFinite(actual[i]) && float.IsFinite(refValues[i]),"Finite values"); error=Math.Max(error,Math.Abs((double)actual[i]-refValues[i])/Math.Max(1,Math.Abs((double)refValues[i]))); }
                    Require(error<=1e-4,"Native scaled error"); values+=actual.LongLength; maximum=Math.Max(maximum,error); held.Add((output,Hash(actual)));
                    observations.Add(new { name,index,repeat,slot,values=actual.Length,sha256=Hash(actual),native_maximum=error,scratch_bytes=expectedScratch,exact=true });
                }
                for (int i = 0; i < tensors.Length; i++) Require(tensors[i] is null || Hash(tensors[i]!.ToArray()) == before[i],"Readonly operand");
                CheckHeld();
            }
            execution.Reset(); CheckHeld();
        }
        Require(observations.Count == 72 && values == 3631104,"All outputs/values");
        Require(!Process.GetCurrentProcess().Modules.Cast<ProcessModule>().Any(m => m.FileName.Contains("onnxruntime",StringComparison.OrdinalIgnoreCase)),"Managed process only");
        File.WriteAllText(args[1],JsonSerializer.Serialize(new { passed=true,core=args[2],role=args[3],width=args[4],flags,vector_count=Vector<float>.Count,
            avx512=Avx512F.IsSupported,hardware_accelerated=Vector.IsHardwareAccelerated,pid=Environment.ProcessId,runtime=Environment.Version.ToString(),
            executable=Sha(typeof(ModelReplay).Assembly.Location),calls=24,distinct_calls=12,outputs=72,values,maximum,observations,
            readonly_operands=true,held_outputs_unchanged=true,no_performance_measurement=true },new JsonSerializerOptions { WriteIndented=true }));
        return 0;
    }
}
