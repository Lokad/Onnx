namespace Lokad.Onnx.CLI;

using System;
using System.Collections.Generic;
using System.IO;

using static Lokad.Onnx.Data;
using static Lokad.Onnx.Text;
using static Lokad.Onnx.Runtime;

#region Enums
public enum ExitResult
{
    SUCCESS = 0,
    UNHANDLED_EXCEPTION = 1,
    INVALID_OPTIONS = 2,
    NOT_FOUND = 4,
    INVALID_INPUT = 5,
    UNKNOWN_ERROR = 7
}
#endregion

class Program
{
    #region Constructor
    static Program()
    {
        AppDomain.CurrentDomain.UnhandledException += Program_UnhandledException;
        Console.CancelKeyPress += Console_CancelKeyPress;
        Console.OutputEncoding = Encoding.UTF8;
    }
    #endregion

    #region Methods

    #region Entry point
    static void Main(string[] args)
    {
        bool debug = (args.Contains("--debug") || args.Contains("-d"));
        UseConsoleLogging(debug);
        var parsed = ArgsParser.Parse(args);
        switch (parsed.Outcome)
        {
            case ParseOutcome.Version:
                Console.WriteLine("Lokad.Onnx v" + AssemblyVersion.ToString(3));
                Exit(ExitResult.SUCCESS);
                return;
            case ParseOutcome.Help:
                if (string.IsNullOrEmpty(parsed.Verb)) PrintGlobalHelp();
                else PrintVerbHelp(parsed.Verb);
                Exit(ExitResult.SUCCESS);
                return;
            case ParseOutcome.Error:
                Error(parsed.Message);
                PrintGlobalHelp();
                Exit(parsed.Exit);
                return;
            default:
                break;
        }
        switch (parsed.Verb)
        {
            case "info":
                if (parsed.Value is InfoOptions info) ShowInfo(info);
                break;
            case "run":
                if (parsed.Value is RunOptions run) Run(run);
                break;
        }
    }
    #endregion

    static void PrintGlobalHelp()
    {
        Console.WriteLine("Lokad.Onnx command-line help");
        Console.WriteLine("Usage: lonnx <command> [options]");
        Console.WriteLine("Commands:");
        Console.WriteLine("  info <file> [--ops] [--init] [--op-filter <type>]   Get information on an ONNX model.");
        Console.WriteLine("  run <file> <inputs...> [options]                    Run an ONNX model or node.");
        Console.WriteLine("Common options:");
        Console.WriteLine("  --debug, -d   Enable debug mode.");
        Console.WriteLine("  --help        Show this help and exit.");
        Console.WriteLine("  --version     Show version and exit.");
    }

    static void PrintVerbHelp(string verb)
    {
        if (verb == "info")
        {
            Console.WriteLine("Lokad.Onnx command-line help");
            Console.WriteLine("Usage: lonnx info <file> [options]");
            Console.WriteLine("Options:");
            Console.WriteLine("  --ops               Only print out a list of distinct ops present in the model.");
            Console.WriteLine("  --init              Only print out a list of initializers present in the model.");
            Console.WriteLine("  --op-filter <type>  Filter on ops with this type.");
        }
        else if (verb == "run")
        {
            Console.WriteLine("Lokad.Onnx command-line help");
            Console.WriteLine("Usage: lonnx run <file> <inputs...> [options]");
            Console.WriteLine("Options:");
            Console.WriteLine("  --save-input        Save any input arguments to the model as additional files.");
            Console.WriteLine("  --softmax           Apply the softmax function to output vectors.");
            Console.WriteLine("  --node <label>      Only run the model node with this label.");
            Console.WriteLine("  --text <model>      Read the user input as text using this model.");
            Console.WriteLine("  --print-input       Print the input tensors that will be fed to the model.");
            Console.WriteLine("  --disable-simd      Disable CPU SIMD features.");
            Console.WriteLine("  --enable-intrinsics Enable CPU SIMD intrinsics.");
            Console.WriteLine("  --profile           Enable the profiler which logs detailed stats about ONNX node execution times.");
            Console.WriteLine("  --optimize-memory   Optimize memory usage at the cost of performance.");
            Console.WriteLine("  --threads <n>       Worker threads for batch-parallel kernels (default 1, sequential).");
        }

    }

    static void ShowInfo(InfoOptions io)
    {
        ExitIfFileNotFound(io.File);
        if (io.Ops)
        {
            PrintModelOps(io.File);
        }
        else if (io.Initializers)
        {
            PrintModelInitializers(io.File);
        }
        else
        {
            PrintModelInfo(io.File, io.OpFilter);
        }
        ExitWithSuccess();
    }
    static void Run(RunOptions ro)
    {
        if (ro.File.StartsWith("http"))
        {
            Error("Remote model locations are not supported; download the model first and pass a local file path.");
            Exit(ExitResult.NOT_FOUND);
            return;
        }
        ExitIfFileNotFound(ro.File);

        var graph = OnnxImport.Load(ro.File);
        if (graph is null)
        {
            Exit(ExitResult.INVALID_INPUT);
            return;
        }
        ITensor[]? ui;
        if (!string.IsNullOrEmpty(ro.Text))
        {
            if (ro.Inputs.Count() != 1)
            {
                Error("The --text option requires exactly one input argument.");
                Exit(ExitResult.INVALID_INPUT);
                return;
            }
            var head = ro.Text.Split(':')[0];
            if ((string.IsNullOrEmpty(head) || head == "me5s") && !Lokad.Onnx.Text.EnsureMe5sTokenizer())
            {
                Exit(ExitResult.INVALID_INPUT);
                return;
            }
            ui = GetTextTensors(ro.Inputs.First(), ro.Text);
        }
        else
        {
            ui = GetInputTensorsFromFileArgs(ro.Inputs, ro.SaveInput);
        }
        if (ui is null || ui.Length == 0)
        {
            Exit(ExitResult.INVALID_INPUT);
            return;
        }
        if (ro.PrintInput)
        {
            Info("Printing {c} input tensor(s)...", ui.Length);
            foreach (var t in ui)
            {
                Info("{n}:{d}", t.TensorNameDesc(), t.PrintData(false));
            }
        }

        bool useSimd = !ro.DisableSimd;
        bool useIntrinsics = ro.DisableSimd
            ? false
            : (!System.Numerics.Vector.IsHardwareAccelerated
                ? HardwareIntrinsics.IsX86FmaSupported
                : (ro.EnableIntrinsics || HardwareIntrinsics.IsX86FmaSupported));
        if (ro.Threads < 1)
        {
            Error("Thread count must be at least 1.");
            Exit(ExitResult.INVALID_OPTIONS);
            return;
        }
        if (ro.Threads > 1)
        {
            Info("Multi-threaded kernels using up to {t} worker threads.", ro.Threads);
        }
        var execOptions = new ExecutionOptions(
            ro.OptimizeMemory ? OptimizationMode.Memory : OptimizationMode.Speed,
            new TensorExecutionOptions(useSimd, useIntrinsics, ro.Threads));
        if (!useSimd)
        {
            Info("CPU SIMD features disabled.");
        }
        else
        {
            Info("CPU SIMD acceleration: {a}.", System.Numerics.Vector.IsHardwareAccelerated);
            if (System.Numerics.Vector.IsHardwareAccelerated)
            {
                Info("CPU SIMD vector size: {v} bits.", System.Numerics.Vector<int>.Count * 4 * 8);

            }
        }

        if (useIntrinsics)
        {
            Info("CPU SIMD available intrinsics: {s}.", HardwareIntrinsics.GetFullInfo());
        }
        else
        {
            Info("Not using CPU SIMD intrinsics.");
        }


        using var profilerScope = ro.EnableProfiler ? Profiler.BeginExecution(true) : null;

        if (ro.Node == "")
        {
            if (graph.Execute(ui, true, ExecutionProvider.CPU, execOptions))
            {
                PrintOutputs(graph, ro);
                if (ro.EnableProfiler && graph.LastProfile is { } profile) PrintProfile(profile);
                Exit(ExitResult.SUCCESS);
            }
            else
            {
                Error("Inference failed: {m}.", graph.LastErrorMessage ?? "invalid inputs");
                Exit(graph.LastFailedNodeName is null ? ExitResult.INVALID_INPUT : ExitResult.UNKNOWN_ERROR);
            }
        }
        else
        {
            if (graph.ExecuteNode(ui, ro.Node, true, ExecutionProvider.CPU, execOptions))
            {
                PrintOutputs(graph, ro);
                Exit(ExitResult.SUCCESS);
            }
            else if (!graph.Nodes.Any(n => n.Name == ro.Node))
            {
                Error("Inference failed: node {n} not found in graph.", ro.Node);
                Exit(ExitResult.NOT_FOUND);
            }
            else
            {
                Error("Inference failed at node {n}: {m}.", ro.Node, graph.LastErrorMessage ?? "invalid inputs");
                Exit(graph.LastFailedNodeName is null ? ExitResult.INVALID_INPUT : ExitResult.UNKNOWN_ERROR);
            }
        }
    }

    static void PrintOutputs(ComputationalGraph graph, RunOptions ro)
    {
        Info("Printing outputs...");
        foreach (var o in graph.Outputs.Values)
        {
            if (o is null) continue;
            if (ro.Softmax && o.Rank == 1)
            {
                Info("Applying softmax to {n}...", o.TensorNameDesc());
                Info("{n}:{v}", o.TensorNameDesc() + "-><softmax>", o.Softmax().PrintData(false));
            }
            else if (ro.Softmax && o is INumericTensor num && num.Rank == 2 && num.Dims[0] == 1)
            {
                Info("Converting {n} to vector and applying softmax...", o.TensorNameDesc());
                Info("{n}:{v}", o.TensorNameDesc() + "-><softmax>", num.RemoveDim(0).Softmax().PrintData(false));
            }
            else
            {
                Info("{n}:{v}", o.TensorNameDesc(), o.PrintData(false));
            }
        }
    }

    static void PrintModelInfo(string file, string? _opfilter)
    {
        ExitIfFileNotFound(file);
        OpType? opfilter = null;
        if (_opfilter is not null)
        {
            if (!Enum.TryParse<OpType>(_opfilter, true, out var op))
            {
                Error("The specified operation type {op} is not valid.", _opfilter);
                Exit(ExitResult.INVALID_OPTIONS);
                return;
            }
            opfilter = op;
        }
        OnnxModel m;
        try
        {
            m = OnnxImport.ParseMetadata(file);
        }
        catch (Exception ex)
        {
            Error(ex, "Could not parse {f} as ONNX model file.", file);
            Exit(ExitResult.INVALID_INPUT);
            return;
        }
        var tensors = new Dictionary<string, string>();
        Info("Graph has input tensors: {i}", m.Inputs.Select(t => t.Describe()));
        Info("Graph has output tensors: {o}", m.Outputs.Select(t => t.Describe()));
        Info("Graph has initializer tensors: {i}", m.Initializers.Select(t => t.Describe()));
        foreach (var t in m.Initializers)
        {
            tensors.Add(t.Name, t.Describe() + "<initializer>");
        }
        foreach (var t in m.Inputs)
        {
            if (!tensors.ContainsKey(t.Name)) tensors.Add(t.Name, t.Describe() + "<input>");
        }
        foreach (var t in m.Outputs)
        {
            tensors.Add(t.Name, t.Describe() + "<output>");
        }
        foreach (var n in m.Nodes)
        {
            foreach (var o in n.Outputs)
            {
                if (!string.IsNullOrEmpty(o) && !tensors.ContainsKey(o)) tensors.Add(o, o + "<intermediate>");
            }
        }
        string GetTensorDesc(string name)
        {
            if (string.IsNullOrEmpty(name))
            {
                return "<empty>";
            }
            return tensors.TryGetValue(name, out var desc) ? desc : $"{name}<unknown>";
        }
        string GetAttributeValueDesc(object value) =>
            value switch
            {
                ITensor i => "tensor " + i.TensorNameDesc() + ":" + i.PrintData(true),
                int n => "int " + n.ToString(),
                int[] na => "int[] " + na.Print(),
                long l => "int64 " + l.ToString(),
                long[] la => "int64[] " + la.Print(),
                float f => "float " + f.ToString(),
                float[] fa => "float[] " + fa.Print(),
                string s => "string " + s,
                string[] sa => "string[] " + sa.Print(),
                _ => throw new NotSupportedException(value.GetType().Name), 
            };
        if (opfilter is null)
        {
            Info("Printing graph nodes...");
        }
        else
        {
            Info("Printing graph nodes with op {op}...", opfilter);
        }
        foreach (var n in m.Nodes)
        {
            OpType op = OpType.Unknown;
            Enum.TryParse<OpType>(n.OpType, false, out op);
            if (opfilter is not null && op != opfilter)
            {
                continue;
            }
            Info("Node {node} has op type: {op}, inputs: {inputs}, outputs: {outputs} and "
                + ((n.Attributes is not null && n.Attributes.Count > 0) ?  "the following attributes:" : "no attributes."),
                n.Name, op.ToString(),
                n.Inputs.Select(t => GetTensorDesc(t)).ToArray(),
                n.Outputs.Select(t => GetTensorDesc(t)).ToArray());

            if (n.Attributes is not null && n.Attributes.Count > 0)
            {
                foreach (var kv in n.Attributes)
                {
                    Info("  {n}: {v}", kv.Key, GetAttributeValueDesc(kv.Value));
                }
            }
        }
    }

    static void PrintModelOps(string file)
    {
        ExitIfFileNotFound(file);
        var m = OnnxImport.ParseMetadata(file);
        Info("Graph has {count} input tensor(s): {in}", m.Inputs.Count, m.Inputs.Select(t => t.Describe()));
        Info("Graph has {count} output tensor(s): {out}", m.Outputs.Count, m.Outputs.Select(t => t.Describe()));
        Info("Graph has {count} initializer tensor(s): {out}", m.Initializers.Count, m.Initializers.Select(t => t.Describe()));
        List<(string Domain, string OpType)> ops = new List<(string Domain, string OpType)>();
        foreach(var node in m.Nodes)
        {
            var key = (node.Domain ?? "", node.OpType ?? "");
            if (!ops.Contains(key))
            {
                ops.Add(key);
            }
        }
        Info("Printing list of distinct ONNX operations in model {f}...", file);
        foreach(var (domain, opName) in ops)
        {
            var display = (string.IsNullOrEmpty(domain) ? "" : domain + ":") + opName + " ";
            int version = m.Opset.TryGetValue(domain ?? "", out var v) ? v
                : m.Opset.TryGetValue("", out var d) ? d : -1;
            bool supported = Enum.TryParse<OpType>(opName, false, out var op)
                && OperatorSchemas.IsSupported(op, domain, version, false);
            Console.WriteLine(display + (supported ? "[supported]" : "[unsupported]"));
        }
        Info("{d} total distinct operations in model. [supported] = supported by backend.", ops.Count);
    }

    static void PrintModelInitializers(string file)
    {
        ExitIfFileNotFound(file);
        var m = OnnxImport.ParseMetadata(file);
        var inputs = m.Inputs.Select(i => i.Name);
        List<string> initializers = new List<string>();
        foreach (var i in m.Initializers)
        {
            if (inputs.Contains(i.Name))
            {
                initializers.Add(i.Describe() + "*");
            }
            else
            {
                initializers.Add(i.Describe());
            }
             
        }
        Info("Graph has {count} input tensors: {in}", m.Inputs.Count, m.Inputs.Select(t => t.Describe()));
        Info("Graph has {count} output tensors: {out}", m.Outputs.Count, m.Outputs.Select(t => t.Describe()));
        Info("Printing list of ONNX initializers in graph...");
        foreach (var i in initializers)
        {
            Console.WriteLine(i);
        }
        Info("{d} total initializers in model. * = initializer for graph input.", m.Initializers.Count);
    }

    static void PrintProfile(Stack<NodeProfile> profile)
    {
        var times = profile.Select(np => (np.Op, np.OpsProfile.Sum(op => op.Time.TotalMilliseconds)))
            .GroupBy(x => x.Item1)
            .Select(g => (g.Key, Convert.ToInt32(g.Sum(gx => gx.Item2)), g.Count()))
            .OrderByDescending(t => t.Item2)
            .ToArray();
        var times2 = profile.Select(np => (np.OpsProfile.Select(op => (op.Stage, op.Time)))).SelectMany(x => x)
         .GroupBy(x => x.Item1)
         .Select(g => (g.Key, Convert.ToInt32(g.Sum(gx => gx.Item2.TotalMilliseconds)), g.Count()))
         .OrderByDescending(t => t.Item2)
         .ToArray();
        Info("Graph op times (ms):");
        foreach (var t in times)
        {
            Info("  {op}({count}): {ms}ms", t.Item1, t.Item3, t.Item2);
        }
        Info("Total graph node count: " + profile.Count);
        Info("Total graph execution time: " + times.Sum(t => t.Item2) + "ms");
        Info("Execution time breakdown (ms):");
        foreach (var t in times2)
        {
            Info("  {stage}: {ms}ms", Profiler.StageDescription(t.Item1), t.Item2);
        }
        var times3 = profile.Select(np => (np.Op,
                                        np.OpsProfile.Select(op => (op.Stage, op.Time.TotalMilliseconds))
                                                        .GroupBy(s => s.Stage)
                                                        .Select(gs => (gs.Key, gs.Sum(i => i.TotalMilliseconds)))))
            .GroupBy(x => x.Item1)
            .Select(x => (x.Key, x.Select(i => i.Item2)
                                    .SelectMany(x => x)
                                    .GroupBy(x => x.Key)
                                    .Select(x => (x.Key, x.Sum(i => i.Item2)))))
            .ToArray();
        Info("Per-op stage breakdown (ms):");
        foreach (var op in times3)
        {
            Info("  {op}: {stages}", op.Item1, string.Join(", ", op.Item2.Select(s => Profiler.StageDescription(s.Item1) + "=" + s.Item2.ToString("F1") + "ms")));
        }
    }

    public static void Exit(ExitResult result)
    {
        if (Cts != null && !Cts.Token.CanBeCanceled)
        {
            Cts.Cancel();
            Cts.Dispose();
        }
        Environment.Exit((int)result);
    }

    public static void ExitIfFileNotFound(string filePath)
    {
        if (filePath.StartsWith("http://") || filePath.StartsWith("https://")) return;
        if (!File.Exists(filePath))
        {
            Error("The file {0} does not exist.", filePath);
            Exit(ExitResult.NOT_FOUND);
        }
    }

    public static void ExitWithSuccess() => Exit(ExitResult.SUCCESS);

    public static void UseConsoleLogging(bool debug)
    {
        Log.MinLevel = debug ? Lokad.Onnx.LogLevel.Debug : Lokad.Onnx.LogLevel.Info;
        Log.Sink = (level, message) =>
        {
            var stamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.ffff");
            Console.WriteLine(stamp + " " + level.ToString().ToUpperInvariant().PadRight(5) + " " + message);
        };
    }

    #endregion

    #region Event Handlers
    private static void Program_UnhandledException(object sender, UnhandledExceptionEventArgs e)
    {
        Error("Unhandled runtime error occurred. Lokad.Onnx CLI will now shutdown.");
        Console.Error.WriteLine((e.ExceptionObject as Exception)?.ToString() ?? "Unknown error.");
        Exit(ExitResult.UNHANDLED_EXCEPTION);
    }

    private static void Console_CancelKeyPress(object? sender, ConsoleCancelEventArgs e)
    {
        Info("Ctrl-C pressed. Exiting.");
        Cts.Cancel();
        Exit(ExitResult.SUCCESS);
    }
    #endregion
    
    #region Fields
    static readonly CancellationTokenSource Cts = new CancellationTokenSource();
    #endregion
}
