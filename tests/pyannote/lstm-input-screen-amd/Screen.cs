using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

internal static class Screen
{
    static void Require(bool condition, string message) { if (!condition) throw new InvalidDataException(message); }
    static string Sha(string path) { using var f = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(f)); }
    static string Hash(DenseTensor<float> tensor)
    {
        Require(!tensor.IsReversedStride && tensor.Buffer.Length == tensor.Length, "Contiguous owned tensor");
        return Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(tensor.Buffer.Span)));
    }
    static DenseTensor<float> Load(string directory, JsonElement item)
    {
        string path = Path.Combine(directory, item.GetProperty("file").GetString()!);
        Require(Sha(path) == item.GetProperty("sha256").GetString() && new FileInfo(path).Length == item.GetProperty("bytes").GetInt64(), "Tensor identity");
        return new DenseTensor<float>(MemoryMarshal.Cast<byte,float>(File.ReadAllBytes(path)).ToArray(),
            item.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32()).ToArray());
    }
    sealed record Case(int Ordinal, string Name, int Index, int Repeats, GraphExecution Execution,
        Dictionary<string,ITensor> Inputs, DenseTensor<float>?[] Tensors, string?[] Before,
        string[] OutputNames, JsonElement[] Outputs, DenseTensor<float>[] Native, long Scratch);

    static int Main(string[] args)
    {
        Require(args.Length == 5, "fixtures output-directory expected-core role qualify-or-time");
        Require(OperatingSystem.IsLinux() && Environment.ProcessorCount == 1 && Process.GetCurrentProcess().ProcessorAffinity.ToInt64() == 4, "AMD CPU2 before CLR");
        Require(args[3] is "selected" or "candidate" && args[4] is "qualify" or "time", "Modes");
        Require(Sha(typeof(CPUExecutionProvider).Assembly.Location) == args[2], "Core identity");
        var flags = Environment.GetEnvironmentVariables().Keys.Cast<string>().Where(k => k.StartsWith("DOTNET_",StringComparison.OrdinalIgnoreCase)
            || k.StartsWith("COMPlus_",StringComparison.OrdinalIgnoreCase) || k.StartsWith("LOKAD_",StringComparison.OrdinalIgnoreCase)).ToArray();
        Require(flags.Length == 0 && Avx512F.IsSupported && Vector.IsHardwareAccelerated && Vector<float>.Count == 8, "Ordinary AMD route");
        string destination = args[1], resultPath = Path.Combine(destination,"result.json");
        Require(!File.Exists(resultPath), "Fresh destination");
        using var log = new StreamWriter(new FileStream(Path.Combine(destination,"events.jsonl"),FileMode.CreateNew));
        void Write(object value) { log.WriteLine(JsonSerializer.Serialize(value)); log.Flush(); }
        bool passed = false; string? failure = null; int clocks = 0, verified = 0, preparations = 0;
        long values = 0; double maximum = 0;
        var held = new List<(DenseTensor<float> Tensor, string Hash)>();
        void CheckHeld() { foreach (var item in held) Require(Hash(item.Tensor) == item.Hash,"Held output changed"); }
        try
        {
            string folder = Path.Combine(args[0],"output"), nativeFolder = Path.Combine(args[0],"native");
            using var capture = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(folder,"result.json")));
            using var native = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(nativeFolder,"result.json")));
            Require(capture.RootElement.GetProperty("passed").GetBoolean() && native.RootElement.GetProperty("passed").GetBoolean(), "Qualified references");
            var calls = capture.RootElement.GetProperty("calls").EnumerateArray().ToArray();
            var references = native.RootElement.GetProperty("reports").EnumerateArray().ToArray();
            Require(calls.Length == 12 && references.Length == 36, "Complete census");
            var cases = new List<Case>();
            for (int ordinal = 0; ordinal < calls.Length; ordinal++)
            {
                var call = calls[ordinal]; int index = call.GetProperty("index").GetInt32(); string name = call.GetProperty("name").GetString()!;
                Require(index == ordinal % 4 && call.GetProperty("opset").GetInt32() == 17, "Original case order");
                var names = call.GetProperty("input_names").EnumerateArray().Select(v => v.GetString()!).ToArray();
                var outNames = call.GetProperty("output_names").EnumerateArray().Select(v => v.GetString()!).ToArray();
                var tensors = call.GetProperty("inputs").EnumerateArray().Select(v => v.ValueKind == JsonValueKind.Null ? null : Load(folder,v)).ToArray();
                var outputs = call.GetProperty("outputs").EnumerateArray().ToArray();
                Require(names.Length == 7 && names[4] == "" && tensors[4] is null && outNames.Length == 3, "Original optional slots");
                var attrs = call.GetProperty("attributes");
                Require(attrs.EnumerateObject().Count() == 2 && attrs.GetProperty("direction").GetString() == "bidirectional" && attrs.GetProperty("hidden_size").GetInt32() == 128, "Original attributes");
                int inputSize = index == 0 ? 60 : 256;
                Require(tensors[0]!.Dimensions.SequenceEqual(new[]{589,1,inputSize}), "Original X geometry");
                var nativeTensors = new DenseTensor<float>[3];
                for (int slot = 0; slot < 3; slot++)
                {
                    var reference = references[ordinal*3+slot];
                    Require(reference.GetProperty("case").GetString() == name && reference.GetProperty("index").GetInt32() == index && reference.GetProperty("slot").GetInt32() == slot,"Native reference order");
                    nativeTensors[slot] = Load(nativeFolder,reference.GetProperty("reference"));
                    _ = Load(folder,outputs[slot]); // Verify the selected reference bytes as well as their manifest.
                }
                long macs = 2L * 589 * 512 * (inputSize + 128); int repeats = checked((int)(((1L<<31)+macs-1)/macs));
                Require(repeats == (index == 0 ? 19 : 10), "Frozen geometry schedule");
                long start = Stopwatch.GetTimestamp();
                var graph = new ComputationalGraph(32L*1024*1024); graph.Metadata["Name"] = "captured-lstm"; graph.Opset[""] = 17;
                graph.Inputs.Add(names[0],tensors[0]!);
                for (int i=1;i<names.Length;i++) if (tensors[i] is not null) graph.Initializers.Add(names[i],tensors[i]!);
                foreach (string output in outNames) graph.Outputs.Add(output,null);
                graph.Nodes.Add(new Node { ID=ordinal+1,Name=call.GetProperty("node").GetString()!,Op=OpType.LSTM,OpTypeName="LSTM",Domain="",OpsetVersion=17,
                    Inputs=names,Outputs=outNames,Attributes=new() { ["direction"]="bidirectional",["hidden_size"]=128 } });
                graph.Prepare(); var execution = graph.CreateExecution(ExecutionOptions.Memory);
                long end = Stopwatch.GetTimestamp();
                Write(new { kind="prepare",ordinal,start,end,frequency=Stopwatch.Frequency }); preparations++;
                cases.Add(new Case(ordinal,name,index,repeats,execution,new Dictionary<string,ITensor>{{names[0],tensors[0]!}},tensors,
                    tensors.Select(t => t is null ? null : Hash(t)).ToArray(),outNames,outputs,nativeTensors,
                    (tensors[1]!.Length+tensors[2]!.Length)*4L+(args[3]=="candidate"?8192:0)));
            }
            bool timing = args[4] == "time";
            for (int pass = timing ? -1 : 0; pass < (timing ? 3 : 1); pass++)
            foreach (var c in cases)
            for (int repeat = 0; repeat < (timing ? c.Repeats : 2); repeat++)
            {
                CheckHeld(); bool ok = false; long start = Stopwatch.GetTimestamp(), end;
                try { c.Execution.Reset(); ok = c.Execution.Execute(c.Inputs,true,ExecutionProvider.CPU,ExecutionOptions.Memory); }
                finally
                {
                    end = Stopwatch.GetTimestamp();
                    Write(new { kind="call",ordinal=c.Ordinal,pass,repeat,start,end,frequency=Stopwatch.Frequency,ok }); clocks++;
                }
                Require(ok,c.Execution.LastErrorMessage ?? "LSTM graph");
                Require(c.Execution.LastScratchBytes == c.Scratch,"Exact per-call scratch");
                var hashes = new string[3]; var errors = new double[3]; var counts = new int[3];
                for (int slot=0;slot<3;slot++)
                {
                    var output = (DenseTensor<float>)c.Execution.Outputs[c.OutputNames[slot]]!;
                    Require(output.Dimensions.SequenceEqual(c.Native[slot].Dimensions),"Output shape");
                    hashes[slot]=Hash(output); Require(hashes[slot] == c.Outputs[slot].GetProperty("sha256").GetString(),"Exact selected output");
                    var actual=output.Buffer.Span; var expected=c.Native[slot].Buffer.Span; double error=0;
                    for (int i=0;i<actual.Length;i++)
                    {
                        Require(float.IsFinite(actual[i]) && float.IsFinite(expected[i]),"Finite output/reference");
                        error=Math.Max(error,Math.Abs((double)actual[i]-expected[i])/Math.Max(1,Math.Abs((double)expected[i])));
                    }
                    Require(error<=1e-4,"Native scaled error"); errors[slot]=error; counts[slot]=actual.Length;
                    maximum=Math.Max(maximum,error); values+=actual.Length;
                    if (!timing || pass == -1 && repeat == 0) held.Add((output,hashes[slot]));
                }
                for (int i=0;i<c.Tensors.Length;i++) Require(c.Tensors[i] is null || Hash(c.Tensors[i]!) == c.Before[i],"Readonly operand");
                CheckHeld();
                Write(new { kind="verified",ordinal=c.Ordinal,pass,repeat,hashes,errors,values=counts,scratch=c.Execution.LastScratchBytes,readonly_operands=true,held_outputs_unchanged=true }); verified++;
            }
            foreach (var c in cases) c.Execution.Reset(); CheckHeld();
            Require(preparations == 12 && clocks == (timing?588:24) && verified == clocks && held.Count == (timing?36:72),"Complete census");
            Require(!Process.GetCurrentProcess().Modules.Cast<ProcessModule>().Any(m => m.FileName.Contains("onnxruntime",StringComparison.OrdinalIgnoreCase)),"Managed-only process");
            passed=true;
        }
        catch(Exception e) { failure=e.ToString(); Console.Error.WriteLine(failure); }
        finally
        {
            log.Dispose();
            File.WriteAllText(resultPath,JsonSerializer.Serialize(new { passed,failure,core=args[2],role=args[3],mode=args[4],flags,
                pid=Environment.ProcessId,runtime=Environment.Version.ToString(),vector_count=Vector<float>.Count,avx512=Avx512F.IsSupported,
                executable=Sha(typeof(Screen).Assembly.Location),frequency=Stopwatch.Frequency,preparations,clocks,verified,values,maximum,
                held=held.Count,events=Sha(Path.Combine(destination,"events.jsonl")) },new JsonSerializerOptions { WriteIndented=true }));
        }
        return passed?0:1;
    }
}
