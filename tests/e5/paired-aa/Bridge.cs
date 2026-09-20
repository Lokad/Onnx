using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Lokad.Onnx;

// Loaded twice into separate assembly contexts. Only framework types cross the boundary.
public static class PairedBridge
{
    static ComputationalGraph graph = null!;
    static Dictionary<string, ITensor> inputs = null!;
    static Dictionary<string, long[]> inputValues = null!;
    static Tensor<float> held = null!;
    static byte[] heldBits = null!;
    static string destination = "", inputHash = "", name = "", referenceHash = "";
    static int[] expectedShape = null!;
    static float[] expected = null!;
    static double beforeError;
    static long loadTicks;
    static (long Execute, long Request, long Bytes, int G0, int G1, int G2) first;
    static readonly ExecutionOptions Options = ExecutionOptions.Memory;

    static void Require(bool ok, string message) { if (!ok) throw new InvalidDataException(message); }
    public static string HashFile(string file)
    {
        using var stream = File.OpenRead(file);
        return Convert.ToHexStringLower(SHA256.HashData(stream));
    }

    static string HashInputs()
    {
        using var memory = new MemoryStream();
        using (var writer = new BinaryWriter(memory, Encoding.UTF8, true))
        {
            writer.Write(Encoding.ASCII.GetBytes("LOKAD-CAMPAIGN-INPUTS-1\0"));
            writer.Write(inputs.Count);
            foreach (string key in inputs.Keys.Order(StringComparer.Ordinal))
            {
                byte[] bytes = Encoding.UTF8.GetBytes(key); writer.Write(bytes.Length); writer.Write(bytes);
                var tensor = (Tensor<long>)inputs[key];
                writer.Write(7); writer.Write(tensor.Dimensions.Length);
                foreach(int dimension in tensor.Dimensions)writer.Write(dimension);
                writer.Write((long)tensor.Length);
                foreach (long value in tensor.ToArray()) writer.Write(value);
            }
        }
        return Convert.ToHexStringLower(SHA256.HashData(memory.ToArray()));
    }

    public static string Initialize(string model, string caseFile, string output)
    {
        Require(graph is null && !Directory.Exists(output), "Bridge initialized or output exists");
        Require(Environment.ProcessorCount == 1 && Options.Tensor.MaxDegreeOfParallelism == 1, "One CPU required before startup");
        destination = output; Directory.CreateDirectory(output);
        using var document = JsonDocument.Parse(File.ReadAllBytes(caseFile));
        var fixture = document.RootElement;
        Require(HashFile(model) == fixture.GetProperty("model_sha256").GetString(), "Model hash");
        name = fixture.GetProperty("name").GetString()!;
        referenceHash = fixture.GetProperty("reference_sha256").GetString()!;
        string reference = Path.Combine(Path.GetDirectoryName(caseFile)!, fixture.GetProperty("reference_file").GetString()!);
        Require(HashFile(reference) == referenceHash, "Reference bytes");
        expectedShape = fixture.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32()).ToArray();
        expected = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(reference)).ToArray();
        Require(expectedShape.Length == 3 && expectedShape[0] == 1 && expectedShape[2] == 384
            && expected.Length == expectedShape.Aggregate(1, (x,y) => checked(x*y)) && expected.All(float.IsFinite), "Reference geometry/values");
        inputValues = fixture.GetProperty("inputs").EnumerateObject().ToDictionary(p => p.Name,
            p => p.Value.EnumerateArray().Select(v => v.GetInt64()).ToArray(), StringComparer.Ordinal);
        Require(inputValues.Keys.Order(StringComparer.Ordinal).SequenceEqual(new[] { "attention_mask", "input_ids", "token_type_ids" }), "Input names");
        Require(inputValues.Values.All(v => v.Length == expectedShape[1]), "Input geometry");
        inputs = inputValues.ToDictionary(p => p.Key, p => (ITensor)new DenseTensor<long>(p.Value.ToArray(), new[] { 1, p.Value.Length }), StringComparer.Ordinal);
        inputHash = HashInputs(); Require(inputHash == fixture.GetProperty("input_sha256").GetString(), "Canonical input hash");
        long start = Stopwatch.GetTimestamp();
        graph = OnnxImport.Load(model) ?? throw new InvalidDataException("Model load");
        loadTicks = Stopwatch.GetTimestamp() - start;
        Require(graph.OutputDescs.Select(v => v.Name).SequenceEqual(new[] { "last_hidden_state" }), "Output names");
        first = Run(); held = (Tensor<float>)graph.Outputs["last_hidden_state"];
        heldBits = MemoryMarshal.AsBytes(held.ToArray().AsSpan()).ToArray();
        beforeError = SaveAndCheck("before");
        return JsonSerializer.Serialize(new { name, input_sha256=inputHash, inputs=inputValues, load_ticks=loadTicks,
            first=new { execute=first.Execute, request=first.Request, bytes=first.Bytes, gc=new[]{first.G0,first.G1,first.G2} },
            core_sha256=HashFile(typeof(ComputationalGraph).Assembly.Location), bridge_sha256=HashFile(typeof(PairedBridge).Assembly.Location),
            reference_sha256=referenceHash, before_error=beforeError, shape=expectedShape, options="Memory" });
    }

    public static (long Execute, long Request, long Bytes, int G0, int G1, int G2) Run()
    {
        int g0=GC.CollectionCount(0), g1=GC.CollectionCount(1), g2=GC.CollectionCount(2);
        long allocated=GC.GetTotalAllocatedBytes(true), request=Stopwatch.GetTimestamp();
        graph.Reset();
        long start=Stopwatch.GetTimestamp();
        bool success=graph.Execute(inputs, true, ExecutionProvider.CPU, Options);
        long end=Stopwatch.GetTimestamp();
        Require(success, graph.LastErrorMessage ?? "Execute failed");
        return (end-start, end-request, GC.GetTotalAllocatedBytes(true)-allocated,
            GC.CollectionCount(0)-g0, GC.CollectionCount(1)-g1, GC.CollectionCount(2)-g2);
    }

    static double SaveAndCheck(string stage)
    {
        var tensor = (Tensor<float>)graph.Outputs["last_hidden_state"];
        Require(tensor.Dimensions.SequenceEqual(expectedShape), "Output shape");
        var actual=tensor.ToArray(); double error=0;
        for(int i=0;i<actual.Length;i++)
        {
            Require(float.IsFinite(actual[i]), "Nonfinite output");
            error=Math.Max(error, Math.Abs((double)actual[i]-expected[i])/Math.Max(1,Math.Abs((double)expected[i])));
        }
        Require(error<=1e-4, "Native numerical gate");
        var bits=MemoryMarshal.AsBytes(actual.AsSpan()).ToArray();
        Require(bits.AsSpan().SequenceEqual(heldBits), "Output bits changed");
        using var stream = new FileStream(Path.Combine(destination, stage+".f32"), FileMode.CreateNew);
        stream.Write(bits); return error;
    }

    public static string Finish()
    {
        Require(HashInputs()==inputHash, "Inputs changed");
        Require(MemoryMarshal.AsBytes(held.ToArray().AsSpan()).SequenceEqual(heldBits), "Held output changed");
        double afterError=SaveAndCheck("after");
        return JsonSerializer.Serialize(new { name, after_error=afterError, inputs_unchanged=true, held_outputs_unchanged=true,
            output_sha256=HashFile(Path.Combine(destination,"after.f32")), before_sha256=HashFile(Path.Combine(destination,"before.f32")) });
    }
}
