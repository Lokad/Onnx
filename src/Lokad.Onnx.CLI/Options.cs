namespace Lokad.Onnx.CLI;

using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;

using CommandLine;
using CommandLine.Text;

#region Base classes
public class Options
{
    [Option("debug", Required = false, HelpText = "Enable debug mode.")]
    public bool Debug { get; set; }

    [Option("options", Required = false, HelpText = "Any additional options for the selected operation.")]
    public string AdditionalOptions { get; set; } = String.Empty;

    public static Dictionary<string, object> Parse(string o)
    {
        Dictionary<string, object> options = new Dictionary<string, object>();
        Regex re = new Regex(@"(\w+)\=([^\,]+)", RegexOptions.Compiled);
        string[] pairs = o.Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
        foreach (string s in pairs)
        {
            Match m = re.Match(s);
            if (!m.Success)
            {
                options.Add("_ERROR_", s);
            }
            else if (options.ContainsKey(m.Groups[1].Value))
            {
                options[m.Groups[1].Value] = m.Groups[2].Value;
            }
            else
            {
                options.Add(m.Groups[1].Value, m.Groups[2].Value);
            }
        }
        return options;
    }
}
#endregion

[Verb("info", HelpText = "Get information on an ONNX model.")]
public class InfoOptions : Options
{
    [Value(0, Required = true, HelpText = "The ONNX model file to open.")]
    public string File { get; set; } = String.Empty;

    [Option("ops", Required = false, HelpText = "Only print out a list of distinct ops present in the model.")]
    public bool Ops { get; set; }

    [Option("init", Required = false, HelpText = "Only print out a list of initializers present in the model.")]
    public bool Initializers { get; set; }

    [Option("op-filter", Required = false, HelpText = "Filter on ops with this type.")]
    public string? OpFilter { get; set; }
}

[Verb("run", HelpText = "Run an ONNX model or node.")]
public class RunOptions : Options
{
    [Value(0, Required = true, HelpText = "The ONNX model file to open.")]
    public string File { get; set; } = String.Empty;

    [Value(1, Required = true, HelpText = "The user input arguments to the model.")]
    public IEnumerable<string> Inputs { get; set; } = Array.Empty<string>();

    [Option("save-input", Required = false, HelpText = "Save any input arguments to the model as additional files.")]
    public bool SaveInput { get; set; }

    [Option("softmax", Required = false, HelpText = "Apply the softmax function to output vectors.")]
    public bool Softmax { get; set; }

    [Option("op-times", Required = false, HelpText = "After every n ops, print the time spent executing each op type.", Default =-1)]
    public int OpTimes { get; set; }

    [Option("node", Required = false, HelpText = "Only run the model node with this label. The specified user inputs together with the graph initializers will be used as the node inputs.")]
    public string Node { get; set; } = "";

    [Option("text", Required = false, HelpText = "The specified user input should be read as text using this model.")]
    public string Text { get; set; } = "";

    [Option("print-input", Required = false, HelpText = "Print the input tensors that will be fed to the model.")]
    public bool PrintInput { get; set; }

    [Option("disable-simd", Required = false, HelpText = "Disable CPU SIMD features.")]
    public bool DisableSimd { get; set; }

    [Option("enable-intrinsics", Required = false, HelpText = "Enable CPU SIMD intrinsics.")]
    public bool EnableIntrinsics { get; set; }

    [Option("profile", Required = false, HelpText = "Enable the profiler which logs detailed stats about ONNX node execution times.")]
    public bool EnableProfiler { get; set; }

    [Option("optimize-memory", Required = false, HelpText = "Optimize memory usage at the cost of performance.")]
    public bool OptimizeMemory { get; set; }
}

[Verb("benchmark", HelpText = "Benchmark an ONNX model or operations.")]
public class BenchmarkOptions : Options
{
    [Value(1, Required = true, HelpText = "The benchmark to run. Currently supported: matmul2d, matmul, indexing, me5s-load, me5s-run")]
    public string BenchmarkId { get; set; } = "";

    [Option('f', "filter", Required = false, HelpText = "Filter the benchmarks by their full name (namespace.typeName.methodName) using glob patterns.")]
    public string Filter { get; set; } = "";

    [Option("list", Required = false, HelpText = "Allows you to print all of the available benchmark names. Support values are flat or tree.")]
    public string List { get; set; } = ""; //

    [Option("iterationCount", Required = false, HelpText = "How many target iterations should be performed.")]
    public int IterationCount { get; set; }

    [Option("warmupCount", Required = false, HelpText = "How many target iterations should be performed.")]
    public int WarmupCount { get; set; }

    [Option("invocationCount", Required = false, HelpText = "Invocation count in a single iteration.")]
    public int InvocationCount { get; set; }

    [Option("runOncePerIteration", Required = false, HelpText = "Run the benchmark exactly once per iteration.")]
    public int RunOncePerIteration { get; set; }
}