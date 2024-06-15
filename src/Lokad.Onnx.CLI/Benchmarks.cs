namespace Lokad.Onnx.CLI;

using System;
using System.IO;

using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;

using Lokad.Onnx;

[SimpleJob(RuntimeMoniker.Net60)]
internal class MultilingualEmbedded5SmallBenchmarks
{
    [GlobalSetup]
    public void GlobalSetupA() => {}
    ComputationalGraph? graph;

}

internal class Benchmarks : Runtime
{
    internal void RunMe5s()
    {
        Info("Running multilingual-embedded-5-small benchmark...");
        var modelFile = Path.Combine(AssemblyLocation, "benchmark-model.onnx");
        if (!File.Exists(modelFile))
        {
            if (!DownloadFile("benchmark-model.onnx", new Uri("https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/onnx/model.onnx?download=true"), modelFile))
            {
                Error("Could not download benchmark model file.");
                return;
            }
        }
        BenchmarkRunner.Run<MultilingualEmbedded5SmallBenchmarks>();
    }
}
