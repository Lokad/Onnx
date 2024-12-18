﻿namespace Lokad.Onnx.CLI;

using System;
using System.IO;

using CommandLine;
using CommandLine.Text;
using NLog;
using NLog.Config;
using Spectre.Console;

using static Lokad.Onnx.Data;
using static Lokad.Onnx.Text;

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

class Program : Runtime
{
    #region Constructor
    static Program()
    {
        AppDomain.CurrentDomain.UnhandledException += Program_UnhandledException;
        InteractiveConsole = true;
        Console.CancelKeyPress += Console_CancelKeyPress;
        Console.OutputEncoding = Encoding.UTF8;
        foreach (var t in optionTypes)
        {
            optionTypesMap.Add(t.Name, t);
        }
    }
    #endregion

    #region Methods

    #region Entry point
    static void Main(string[] args)
    {
        string toolname = "Lokad.Onnx.CLI";
        string logname = "CLI";
        bool debug = (args.Contains("--debug") || args.Contains("-d"));
        Initialize(toolname, logname, debug, CreateConsoleLogger(debug, logname, true));
        PrintLogo();
        var result = new Parser().ParseArguments(args, optionTypes);
        result
            .WithParsed<InfoOptions>(Info)
            .WithParsed<RunOptions>(Run)
            .WithParsed<BenchmarkOptions>(bo => Benchmark(bo, GetBenchmarkArgs(args, bo)))
            .WithNotParsed(errors => Help(result, errors));
    }
    #endregion

    static void Help(ParserResult<object> result, IEnumerable<Error> errors)
    {
        HelpText help = GetAutoBuiltHelpText(result);
        help.Heading = new HeadingInfo("Lokad.Onnx command-line help");
        help.Copyright = "";
        if (errors.Any(e => e.Tag == ErrorType.VersionRequestedError))
        {
            help.Heading = new HeadingInfo("Lokad.Onnx", AssemblyVersion.ToString(3));
            help.Copyright = "";
            Info(help);
            Exit(ExitResult.SUCCESS);
        }
        else if (errors.Any(e => e.Tag == ErrorType.HelpVerbRequestedError))
        {
            HelpVerbRequestedError error = (HelpVerbRequestedError)errors.First(e => e.Tag == ErrorType.HelpVerbRequestedError);
            if (error.Type != null)
            {
                help.AddVerbs(error.Type);
            }
            else
            {
                help.AddVerbs(optionTypes);
            }
            Info(help.ToString().Replace("--", ""));
            Exit(ExitResult.SUCCESS);
        }
        else if (errors.Any(e => e.Tag == ErrorType.HelpRequestedError))
        {
            HelpRequestedError error = (HelpRequestedError)errors.First(e => e.Tag == ErrorType.HelpRequestedError);
            help.AddVerbs(result.TypeInfo.Current);
            help.AddOptions(result);
            help.AddPreOptionsLine($"{result.TypeInfo.Current.Name.Replace("Options", "").ToLower()} options:");
            Info(help);
            Exit(ExitResult.SUCCESS);
        }
        else if (errors.Any(e => e.Tag == ErrorType.NoVerbSelectedError))
        {
            help.AddVerbs(optionTypes);
            Info(help);
            Exit(ExitResult.INVALID_OPTIONS);
        }
        else if (errors.Any(e => e.Tag == ErrorType.MissingRequiredOptionError))
        {
            MissingRequiredOptionError error = (MissingRequiredOptionError)errors.First(e => e.Tag == ErrorType.MissingRequiredOptionError);
            Info(help);
            Error("A required option is missing.");

            Exit(ExitResult.INVALID_OPTIONS);
        }
        else if (errors.Any(e => e.Tag == ErrorType.UnknownOptionError))
        {
            UnknownOptionError error = (UnknownOptionError)errors.First(e => e.Tag == ErrorType.UnknownOptionError);
            help.AddVerbs(optionTypes);
            Info(help);
            Error("Unknown option: {error}.", error.Token);
            Exit(ExitResult.INVALID_OPTIONS);
        }
        else
        {
            Error("An error occurred parsing the program options: {errors}.", errors);
            help.AddVerbs(optionTypes);
            Info(help);
            Exit(ExitResult.INVALID_OPTIONS);
        }
    }

    static void Info(InfoOptions io)
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
            if (Uri.TryCreate(ro.File, UriKind.Absolute, out Uri? uri) && DownloadFile("ONNX model file", uri, Path.Combine(Directory.GetCurrentDirectory(), "model.onnx")))
            {
                ro.File = Path.Combine(Directory.GetCurrentDirectory(), "model.onnx");
                Info("Successfully downloaded model file.");
            }
            else
            {
                Error("Could not download model file.");
                Exit(ExitResult.NOT_FOUND);
            }
        }
        else
        {
            ExitIfFileNotFound(ro.File);
        }

        var graph = Model.Load(ro.File);
        if (graph is null)
        {
            Exit(ExitResult.INVALID_INPUT);
            return;
        }
        ITensor[]? ui;
        if (!string.IsNullOrEmpty(ro.Text))
        {
            ui = GetTextTensors(ro.Inputs.First(), ro.Text);
        }
        else
        {
            ui = GetInputTensorsFromFileArgs(ro.Inputs);
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

        if (ro.DisableSimd)
        {
            HardwareConfig.UseSimd = false;
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
        
        if (ro.EnableIntrinsics && System.Numerics.Vector.IsHardwareAccelerated)
        {
            HardwareConfig.UseIntrinsics = true;
            Info("CPU SIMD available intrinsics: {s}.", HardwareIntrinsics.GetFullInfo());
        }
        else
        {
            Info("Not using CPU SIMD intrinsics.");
        }

        if (ro.OptimizeMemory)
        {
            CPUExecutionProvider.OptimizationMode = OptimizationMode.Memory;    
        }

        if (ro.EnableProfiler)
        {
            Profiler.Enabled = true;
        }

        if (ro.Node == "")
        {
            if (graph.Execute(ui, true))
            {
                Info("Printing outputs...");
                foreach (var o in graph.Outputs.Values)
                {
                    if (ro.Softmax && o.Rank == 1)
                    {
                        Info("Applying softmax to {n}...", o.TensorNameDesc());
                        Info("{n}:{v}", o.TensorNameDesc() + "-><softmax>", o.Softmax().PrintData(false));
                    }
                    else if (ro.Softmax && o.Rank == 2 && o.Dims[0] == 1)
                    {
                        Info("Converting {n} to vector and applying softmax...", o.TensorNameDesc());
                        Info("{n}:{v}", o.TensorNameDesc() + "-><softmax>", o.RemoveDim(0).Softmax().PrintData(false));
                    }
                    else
                    {
                        Info("{n}:{v}", o.TensorNameDesc(), o.PrintData(false));
                    }
                }
                if (ro.EnableProfiler) PrintProfile();
                Exit(ExitResult.SUCCESS);
            }
        }
        else
        {
            if (graph.ExecuteNode(ui, ro.Node, true))
            {
                Info("Printing outputs...");
                foreach (var o in graph.Outputs.Values)
                {
                    if (ro.Softmax && o.Rank == 1)
                    {
                        Info("Applying softmax to {n}...", o.TensorNameDesc());
                        Info("{n}:{v}", o.TensorNameDesc() + "-><softmax>", o.Softmax().PrintData(false));
                    }
                    else if (ro.Softmax && o.Rank == 2 && o.Dims[0] == 1)
                    {
                        Info("Converting {n} to vector and applying softmax...", o.TensorNameDesc());
                        Info("{n}:{v}", o.TensorNameDesc() + "-><softmax>", o.RemoveDim(0).Softmax().PrintData(false));
                    }
                    else
                    {
                        Info("{n}:{v}", o.TensorNameDesc(), o.PrintData(false));
                    }
                }
                Exit(ExitResult.SUCCESS);
            }
        }
    }

    static void Benchmark(BenchmarkOptions bo, string[] args)
    {
        try
        {
            switch (bo.BenchmarkId)
            {
                case "me5s-load":
                    Benchmarks.RunMe5sLoad(args);
                    ExitWithSuccess();
                    break;
                case "me5s-run":
                    Benchmarks.RunMe5sRun(args);
                    ExitWithSuccess();
                    break;
                case "matmul2d":
                    Benchmarks.RunMatMul2D(args);
                    ExitWithSuccess();
                    break;
                case "matmul":
                    Benchmarks.RunMatMul(args);
                    ExitWithSuccess();
                    break;
                case "indexing":
                    Benchmarks.RunIndexing(args);
                    ExitWithSuccess();
                    break;
                default:
                    Error("Unknown benchmark: {b}.", bo.BenchmarkId);
                    Exit(ExitResult.INVALID_OPTIONS);
                    break;

            }
        }
        catch (InvalidOperationException e)
        {
            if (e.Message == "Sequence contains no elements")
            {
                Exit(ExitResult.SUCCESS);
            }
            else throw new Exception("Exception thrown by BenchmarkDotNet runner.", e);
        }
    }

    static void PrintModelInfo(string file, string? _opfilter = null)
    {
        ExitIfFileNotFound(file);
        OpType? opfilter = null;
        if (_opfilter is not null)
        {
            if (!Enum.TryParse<OpType>(_opfilter, true, out var op))
            {
                Error("The specified operation type {op} is not valid.", _opfilter);
            }
            else
            {
                opfilter = op;  
            }
        }
        var graph = Model.Load(file);
        if (graph is null)
        {
            Exit(ExitResult.INVALID_INPUT);
            return;
        }
        var tensors = new Dictionary<string, string>();
        Info("Graph has input tensors: {i}", graph.Inputs.Select(t => t.Value.TensorNameDesc()));
        Info("Graph has output tensors: {o}", graph.Outputs.Select(t => t.Value.TensorNameDesc()));
        Info("Graph has initializer tensors: {i}", graph.Initializers.Select(t => t.Value.TensorNameDesc()));
        foreach (var t in graph.Initializers.Values)
        {
            tensors.Add(t.Name, t.TensorNameDesc() + "<initializer>");
        }
        foreach (var t in graph.Inputs.Values)
        {
            if (!tensors.ContainsKey(t.Name)) tensors.Add(t.Name, t.TensorNameDesc() + "<input>");
        }
        foreach (var t in graph.Outputs.Values)
        {
            tensors.Add(t.Name, t.TensorNameDesc() + "<output>");
        }
        foreach (var t in graph.IntermediateOutputs)
        {
            tensors.Add(t.Key, t.Key + "<intermediate>");
        }
        if (opfilter is null)
        {
            Info("Printing graph nodes...");
        }
        else
        {
            Info("Printing graph nodes with op {op}...", opfilter);
        }
        foreach (var n in graph.Nodes)
        {
            if (opfilter is not null && n.Op != opfilter)
            {
                continue;
            }
            Info("Node {node} has op type: {op}, inputs: {inputs}, outputs: {outputs} and " 
                + ((n.Attributes is not null && n.Attributes.Count > 0) ?  "the following attributes:" : "no attributes."), 
                n.Name, n.Op.ToString(), 
                n.Inputs.Select(t => tensors[t]).ToArray(), 
                n.Outputs.Select(t => tensors[t]).ToArray());
            
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
        var m = Model.Parse(file);
        if (m is null)
        {
            Exit(ExitResult.INVALID_INPUT);
            return;
        }
        Info("Graph has {count} input tensor(s): {in}", m.Graph.Input.Count, m.Graph.Input.Select(t => t.TensorNameDesc()));
        Info("Graph has {count} output tensor(s): {out}", m.Graph.Output.Count, m.Graph.Output.Select(t => t.TensorNameDesc()));
        Info("Graph has {count} initializer tensor(s): {out}", m.Graph.Initializer.Count, m.Graph.Initializer.Select(t => t.TensorNameDesc()));
        List<OpType> ops = new List<OpType>();
        foreach(var node in m.Graph.Node)
        {
            var op = Enum.Parse<OpType>(node.OpType);
            if (!ops.Contains(op))
            {
                ops.Add(op);
            }
        }
        Info("Printing list of distinct ONNX operations in model {f}...", file);
        foreach(var op in ops)
        {
            if (CPUExecutionProvider.SupportsOp(op))
            {
                Con.Write(new Spectre.Console.Text(op + " ", new Style(foreground: Color.Green)));
            }
            else
            {
                Con.Write(new Spectre.Console.Text(op + " "));
            }
        }
        Con.Write(Environment.NewLine);
        Info("{d} total distinct operations in model. Green = supported by backend.", ops.Count);
    }

    static void PrintModelInitializers(string file)
    {
        ExitIfFileNotFound(file);
        var m = Model.Parse(file);
        if (m is null)
        {
            Exit(ExitResult.INVALID_INPUT);
            return;
        }
        var inputs = m.Graph.Input.Select(i => i.Name);
        List<string> initializers = new List<string>();
        foreach (var i in m.Graph.Initializer)
        {
            if (inputs.Contains(i.Name))
            {
                initializers.Add(i.TensorNameDesc() + "*");
            }
            else
            {
                initializers.Add(i.TensorNameDesc());
            }
             
        }
        Info("Graph has {count} input tensors: {in}", m.Graph.Input.Count, m.Graph.Input.Select(t => t.TensorNameDesc()));
        Info("Graph has {count} output tensors: {out}", m.Graph.Output.Count, m.Graph.Output.Select(t => t.TensorNameDesc()));
        Info("Printing list of ONNX initializers in graph...");
        foreach (var i in initializers)
        {
            Con.WriteLine(i);
        }
        Info("{d} total initializers in model. * = initializer for graph input.", m.Graph.Initializer.Count);
    }

    static void PrintProfile()
    {
        var times = Profiler.Profile.Select(np => (np.Op, np.OpsProfile.Sum(op => op.Time.TotalMilliseconds)))
            .GroupBy(x => x.Item1)
            .Select(g => (g.Key, Convert.ToInt32(g.Sum(gx => gx.Item2)), g.Count()));
        var times2 = Profiler.Profile.Select(np => (np.OpsProfile.Select(op => (op.Stage, op.Time)))).SelectMany(x => x)//Sum(op => op.Time.TotalMilliseconds)))
         .GroupBy(x => x.Item1)
         .Select(g => (g.Key, Convert.ToInt32(g.Sum(gx => gx.Item2.TotalMilliseconds)), g.Count()));
        var chart = new BarChart()
            .Width(100)
            .Label("[green bold underline]Graph Op Times[/]")
            .CenterLabel()
            .AddItems(times, (t) => new BarChartItem($"{t.Item1}({t.Item3})", t.Item2, (Color) (((int) t.Item1 % 10) + 1 )));
        var chart2 = new BreakdownChart()
            .Width(100)
            .Compact()
            .WithValueColor(Color.White)
            .AddItems(times2, t => new BreakdownChartItem(Profiler.StageDescription(t.Item1) + ":", t.Item2, (Color)(((int)t.Item1 % 10) + 1)));
        Con.Write(chart);
        Con.WriteLine();
        Con.WriteLine("Total graph node count: " + Profiler.Profile.Count);
        Con.WriteLine("Total graph execution time: " + times.Sum(t => t.Item2) + "ms");
        Con.Write("Execution time breakdown (ms): ");
        Con.Write(chart2);

        var times3 = Profiler.Profile.Select(np => (np.Op, 
                                        np.OpsProfile.Select(op => (op.Stage, op.Time.TotalMilliseconds))
                                                        .GroupBy(s => s.Stage)
                                                        .Select(gs => (gs.Key, gs.Sum(i => i.TotalMilliseconds)))))
            .GroupBy(x => x.Item1)
            .Select(x => (x.Key, x.Select(i => i.Item2)
                                    .SelectMany(x => x)
                                    .GroupBy(x => x.Key)
                                    .Select(x => (x.Key, x.Sum(i => i.Item2)))))
            .ToArray();

        var grid = new Grid(); 
        grid.AddColumn(); 
        grid.AddColumn();
        grid.AddColumn();

        for(int t = 0; t < times3.Length; t+=3)
        {
            var time = times3[t].Item2.Select(i => new BreakdownChartItem(Profiler.StageDescription(i.Item1), i.Item2, (Color)(((int)i.Item1) + 1)));
            var chart4 = new BreakdownChart()
                .Width(50)
                .Compact()
                .AddItems(time);
            Panel p = new Panel(chart4);
            p.Width = 50;
            p.Header = new PanelHeader(times3[t].Item1.ToString(), Justify.Center);
            if (t + 1 >= times3.Length)
            {
                grid.AddRow(p);
                break;
            }
            else
            {
                var time2 = times3[t + 1].Item2.Select(i => new BreakdownChartItem(Profiler.StageDescription(i.Item1), i.Item2, (Color)(((int)i.Item1) + 1)));
                var chart5 = new BreakdownChart()
                    .Width(50)
                    .Compact()
                    .AddItems(time2);
                var p2 = new Panel(chart5);
                p2.Width = 50;
                p2.Header = new PanelHeader(times3[t + 1].Item1.ToString(), Justify.Center);
                if (t + 2 >= times3.Length)
                {
                    grid.AddRow(p, p2);
                    break;
                }
                else
                {
                    var time3 = times3[t + 2].Item2.Select(i => new BreakdownChartItem(Profiler.StageDescription(i.Item1), i.Item2, (Color)(((int)i.Item1) + 1)));
                    var chart6 = new BreakdownChart()
                        .Width(50)
                        .Compact()
                        .AddItems(time3);
                    var p3 = new Panel(chart6);
                    p3.Width = 50;
                    p3.Header = new PanelHeader(times3[t + 2].Item1.ToString(), Justify.Center);
                    grid.AddRow(p, p2, p3);
                }
            }
        }
        Con.Write(grid);
    }

    static string GetAttributeValueDesc(object value) =>
        value switch
        {
            ITensor i => "tensor " + i.TensorNameDesc() + ":" + i.PrintData(),
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

    static void PrintLogo()
    {
        Con.Write(new FigletText(font, "Lokad.Onnx").Color(Color.Orange1));
        Con.Write(new Spectre.Console.Text($"v{AssemblyVersion.ToString(3)}\n"));
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

    static HelpText GetAutoBuiltHelpText(ParserResult<object> result)
    {
        return HelpText.AutoBuild(result, h =>
        {
            h.AddOptions(result);
            HelpText.DefaultParsingErrorsHandler(result, h);
            return h;
        },
        e => e);
    }

    static string[] GetBenchmarkArgs(string[] args, BenchmarkOptions bo)
    {
        List<string> result = args.ToList();
        result.RemoveRange(0, 2);
        result.Remove("--debug");
        result.Remove("-d");
        
        return result.ToArray();
    }

    public static LoggingConfiguration ConfigureConsoleLogger(LoggingConfiguration config, bool debug, bool color)
    {
        var logconsole = new NLog.Targets.ColoredConsoleTarget("logconsole")
        {
            Layout = debug ? NLogDebugLayout : NLogLayout,
        };
        config.AddTarget(logconsole);
        config.AddRule(new LoggingRule("*", LogLevel.Info, logconsole));
        config.AddRule(new LoggingRule("*", LogLevel.Warn, logconsole));
        config.AddRule(new LoggingRule("*", LogLevel.Error, logconsole));
        config.AddRule(new LoggingRule("*", LogLevel.Fatal, logconsole));
        if (debug)
        {
            config.AddRule(new LoggingRule("*", LogLevel.Debug, logconsole));
        }
        if (color)
        {
            logconsole.RowHighlightingRules.Add(new NLog.Targets.ConsoleRowHighlightingRule() { ForegroundColor = NLog.Targets.ConsoleOutputColor.Gray });

            logconsole.WordHighlightingRules.Add(new NLog.Targets.ConsoleWordHighlightingRule()
            {
                Condition = NLog.Conditions.ConditionParser.ParseExpression("level == LogLevel.Info"),
                Regex = "INFO",
                WholeWords = true,
                CompileRegex = true,
                ForegroundColor = NLog.Targets.ConsoleOutputColor.White,
                BackgroundColor = NLog.Targets.ConsoleOutputColor.DarkGreen
            });

            logconsole.WordHighlightingRules.Add(new NLog.Targets.ConsoleWordHighlightingRule()
            {
                Condition = NLog.Conditions.ConditionParser.ParseExpression("level == LogLevel.Warn"),
                Regex = "WARN",
                WholeWords = true,
                CompileRegex = true,
                ForegroundColor = NLog.Targets.ConsoleOutputColor.White,
                BackgroundColor = NLog.Targets.ConsoleOutputColor.DarkYellow
            });

            logconsole.WordHighlightingRules.Add(new NLog.Targets.ConsoleWordHighlightingRule()
            {
                Condition = NLog.Conditions.ConditionParser.ParseExpression("level == LogLevel.Error"),
                Regex = "ERROR",
                WholeWords = true,
                CompileRegex = true,
                ForegroundColor = NLog.Targets.ConsoleOutputColor.White,
                BackgroundColor = NLog.Targets.ConsoleOutputColor.DarkRed
            });

            logconsole.WordHighlightingRules.Add(new NLog.Targets.ConsoleWordHighlightingRule()
            {
                Condition = NLog.Conditions.ConditionParser.ParseExpression("level == LogLevel.Fatal"),
                Regex = "FATAL",
                WholeWords = true,
                CompileRegex = true,
                ForegroundColor = NLog.Targets.ConsoleOutputColor.White,
                BackgroundColor = NLog.Targets.ConsoleOutputColor.Red
            });

            logconsole.WordHighlightingRules.Add(new NLog.Targets.ConsoleWordHighlightingRule()
            {
                Condition = NLog.Conditions.ConditionParser.ParseExpression("level == LogLevel.Debug"),
                Regex = "DEBUG",
                WholeWords = true,
                CompileRegex = true,
                ForegroundColor = NLog.Targets.ConsoleOutputColor.White,
                BackgroundColor = NLog.Targets.ConsoleOutputColor.DarkBlue
            });


            logconsole.WordHighlightingRules.Add(new NLog.Targets.ConsoleWordHighlightingRule()
            {
                Regex = "\\d\\d\\:\\d\\d\\:\\d\\d\\.\\d{4}",
                CompileRegex = true,
                ForegroundColor = NLog.Targets.ConsoleOutputColor.Gray,
            });
            logconsole.WordHighlightingRules.Add(new NLog.Targets.ConsoleWordHighlightingRule()
            {
                Regex = "\\\"\\S+\\\"",
                CompileRegex = true,
                ForegroundColor = NLog.Targets.ConsoleOutputColor.Red
            });
            logconsole.WordHighlightingRules.Add(new NLog.Targets.ConsoleWordHighlightingRule()
            {
                Regex = "\\s+([-+]?([0-9]*[.])?[0-9]+([eE][-+]?\\\\d+)?(ms)?(?!\\:))",
                CompileRegex = true,
                ForegroundColor = NLog.Targets.ConsoleOutputColor.Cyan,

            });
            logconsole.WordHighlightingRules.Add(new NLog.Targets.ConsoleWordHighlightingRule()
            {
                Regex = "\\[(\\[*)\\s*([-+]?([0-9]*[.])?[0-9]+([eE][-+]?\\d+)?\\,?)*(\\])*\\]",
                CompileRegex = true,
                ForegroundColor = NLog.Targets.ConsoleOutputColor.Magenta,
            });
        }
        return config;
    }

    public static Microsoft.Extensions.Logging.ILogger CreateConsoleLogger(bool debug = false, string logname = "BASE", bool color = false) 
    {
        var config = new LoggingConfiguration();
        if (debug)
        {
            config.Variables["logLevel"] = "Debug";
        }
        LogManager.Configuration = ConfigureConsoleLogger(config, debug, color); ;
        return new NLog.Extensions.Logging.NLogLoggerFactory(
            new NLog.Extensions.Logging.NLogProviderOptions() { 
                AutoShutdown = true,
                ParseMessageTemplates = true,   
                CaptureMessageTemplates = true, 
            }).CreateLogger(logname);
    }
    #endregion

    #region Event Handlers
    private static void Program_UnhandledException(object sender, UnhandledExceptionEventArgs e)
    {
        Error("Unhandled runtime error occurred. Lokad.Onnx CLI will now shutdown.");
        Con.WriteException((Exception)e.ExceptionObject);
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
    static object uilock = new object();
    static Type[] optionTypes =
    {
        typeof(Options), typeof(InfoOptions), typeof(RunOptions), typeof(BenchmarkOptions)

    };
    static FigletFont font = FigletFont.Load(Path.Combine(AssemblyLocation, "chunky.flf"));
    static Dictionary<string, Type> optionTypesMap = new Dictionary<string, Type>();
    static string NLogDebugLayout = "${longdate}${pad:padding=6:inner=${level:uppercase=true}} ${logger}(${threadid}) ${callsite:skipFrames=2:includeNamespace=false}: ${message:withexception=true}";
    static string NLogLayout = "${longdate}${pad:padding=6:inner=${level:uppercase=true}} ${logger}(${threadid}) ${message:withexception=true}";
    #endregion
}