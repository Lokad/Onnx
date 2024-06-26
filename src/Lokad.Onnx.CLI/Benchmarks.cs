namespace Lokad.Onnx.CLI;

using System;
using System.IO;
using System.Linq;
using System.Text.Json;

using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Running;

using static Lokad.Onnx.Text;

[InProcess]
[IterationsColumn]
[MemoryDiagnoser()]
public class MultilingualEmbedded5SmallBenchmarks : Runtime
{
    [Benchmark(Description="1 string of 20 chars")]
    [WarmupCount(3)]
    [IterationCount(3)]
    public void Benchmark20_1() => graph!.Execute(ui20_1!, true);

    [Benchmark(Description = "10 strings of 20 chars")]
    [WarmupCount(1)]
    [IterationCount(3)]
    public void Benchmark20_10() => graph!.Execute(ui20_10!, true);

    [IterationSetup]
    public void Reset() => graph!.Reset();

    [GlobalSetup()]
    public void Setup() 
    {
        var op = Begin("Creating computational graph and tokenizing test data");
        graph = Model.Load(modelFile);
        var options = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true
        };
        TextData?[] textData = File.ReadAllLines(testDataFile).AsParallel().Select(t => JsonSerializer.Deserialize<TextData>(t, options)).ToArray();
        Random rnd = new Random();
        T20 = textData
            .AsParallel()
            .Where(t => t is not null && t!.Text.Length >= 21 && t!.Text[20] == ' ')
            .Select(t => t!.Text.Substring(0, 20)/*.Replace("\n", " ")*/)
            .OrderBy(x => rnd.Next())
            .ToArray();
        T200 = textData
            .AsParallel()
            .Where(t => t is not null && t!.Text.Length >= 201 && t!.Text[200] == ' ')
            .Select(t => t!.Text.Substring(0, 200)/*.Replace("\n", " ")*/)
            .OrderBy(x => rnd.Next())
            .ToArray();

        ui20_1 = GetTextTensors(T20[0], "me5s");
        ui20_10 = GetTextTensors(T20[1..11], "me5s");
        ui20_100 = GetTextTensors(T20[11..111], "me5s");
        op.Complete();  
    }

    #region Fields
    string modelFile = Path.Combine(Runtime.AssemblyLocation, "benchmark-model.onnx");
    string testDataFile = Path.Combine(Runtime.AssemblyLocation, "train.jsonl");
    public static string[] T20 = Array.Empty<string>();
    public static string[] T200 = Array.Empty<string>();
    public static ComputationalGraph? graph;
    ITensor[]? ui20_1 = null;
    ITensor[]? ui20_10 = null;
    ITensor[]? ui20_100 = null;
    #endregion
}

internal class Benchmarks : Runtime
{
    internal static void RunMe5s()
    {
        var op = Begin("Preparing model and data for multilingual-embedded-5-small benchmark");
        var modelFile = Path.Combine(AssemblyLocation, "benchmark-model.onnx");
        var testDataFile = Path.Combine(AssemblyLocation, "train.jsonl");
        if (!File.Exists(modelFile))
        {
            if (!DownloadFile("benchmark-model.onnx", new Uri("https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/onnx/model.onnx?download=true"), modelFile))
            {
                Error("Could not download benchmark model file.");
                op.Abandon();
                return;
            }
        }

        if (!File.Exists(testDataFile))//
        {  //https://huggingface.co/datasets/mteb/quora/resolve/main/corpus.jsonl
            if (!DownloadFile("train.jsonl", new Uri("https://huggingface.co/datasets/mteb/amazon_reviews_multi/resolve/main/en/train.jsonl?download=true"), testDataFile))
            {
                Error("Could not download benchmark test data file.");
                op.Abandon();
                return;
            }
        }
        op.Complete();
        BenchmarkRunner.Run<MultilingualEmbedded5SmallBenchmarks>();
    }
}

public partial class TextData
{
    public string Id { get; set; } = "";

    public string Text { get; set; } = "";

    public long Label { get; set; } = 0;

    public string Label_Text { get; set; } = "";
}

public class BenchmarkConfig : ManualConfig
{
}