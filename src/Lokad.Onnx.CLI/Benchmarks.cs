namespace Lokad.Onnx.CLI;

using System;
using System.IO;
using System.Reflection.Metadata;
using System.Text.Json;

using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;

using Lokad.Onnx;

[SimpleJob(RuntimeMoniker.Net60)]
public class MultilingualEmbedded5SmallBenchmarks
{
    [GlobalSetup]
    public void GlobalSetup()
    {
        graph = Model.Load(modelFile);
    }

    [Benchmark]
    public void Ten()
    {

    }

    [GlobalSetup]
    public void SetupTen() 
    {
        ui10 = Text.GetTextTensors("me5s",";;");
    }
    static string modelFile = Path.Combine(Runtime.AssemblyLocation, "benchmark-model.onnx");
    ComputationalGraph? graph;
    ITensor[]? ui10;
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
        var options = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true
        };
        TextData?[] textData = File.ReadAllLines(testDataFile).AsParallel().Select(t => JsonSerializer.Deserialize<TextData>(t, options)).ToArray();
        Random rnd = new Random();
        T20 = textData
            .AsParallel()
            .Where(t => t is not null)
            .Select(t => t!.Text/*.Replace("\n", " ")*/)
            .Where(t => t.Length >= 21 && t[20] == ' ')
            .Select(t => t.Substring(0, 20))
            .ToArray();
        T200 = textData
            .AsParallel()
            .Where(t => t is not null)
            .Select(t => t!.Text/*.Replace("\n", " ")*/)
            .Where(t => t.Length >= 201 && t[200] == ' ')
            .Select(t => t.Substring(0, 200))
            .ToArray();
        op.Complete();
        BenchmarkRunner.Run<MultilingualEmbedded5SmallBenchmarks>();
    }

    public static string[] T20 = Array.Empty<string>();
    public static string[] T200 = Array.Empty<string>();

}



    public partial class TextData
{
    public string Id { get; set; } = "";

    public string Text { get; set; } = "";

    public long Label { get; set; } = 0;

    public string Label_Text { get; set; } = "";
}
