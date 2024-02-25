namespace Lokad.Onnx.CLI;

using System;
using System.IO;
using System.Globalization;
using System.Reflection;
using System.Runtime.Versioning;
using CommandLine;
using CommandLine.Text;
using Lokad.Onnx;
using Lokad.Onnx.Backend;

//using Satsuma;

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
    [RequiresPreviewFeatures]
    static void Main(string[] args)
    {
        Initialize("Lokad.Onnx.CLI", "CLI", (args.Contains("--debug") || args.Contains("-d")), true, true);
        PrintLogo();
        var result = new Parser().ParseArguments<Options, InfoOptions, RunOptions>(args);
        result.WithNotParsed(errors =>
        {
            HelpText help = GetAutoBuiltHelpText(result);
            help.Heading = new HeadingInfo("Lokad.Onnx", AssemblyVersion.ToString(3));
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
        })
        .WithParsed<InfoOptions>(io =>
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
                PrintModelInfo(io.File, io.FilterOp); 
            }
        })
        .WithParsed<RunOptions>(ro =>
        {
            Run(ro.File, ro.Inputs);
        });
    }
    #endregion

    [RequiresPreviewFeatures]
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
        var graph = Model.LoadFromFile(file);
        if (graph is null)
        {
            Exit(ExitResult.INVALID_INPUT);
            return;
        }
       
        Info("Graph details: Name: {name}. Domain: {dom}. Producer name: {pn}. Producer version: {pv}. IR Version: {ir}. DocString: {ds}.", graph.Metadata["Name"], graph.Metadata["Domain"], graph.Metadata["ProducerName"], graph.Metadata["ProducerVersion"], graph.Metadata["IrVersion"]?.ToString() ??"", graph.Metadata["DocString"]);
  
        var tensors = new Dictionary<string, string>();
        Info($"Graph has input tensors: {{{graph.Inputs.Select(t => t.Value.TensorNameDesc()).JoinWith(",")}}}");
        Info($"Graph has output tensors: {{{graph.Outputs.Select(t => t.Value.TensorNameDesc()).JoinWith(",")}}}");
        Info($"Graph has initializer tensors: {{{graph.Initializers.Select(t => t.Value.TensorNameDesc()).JoinWith(",")}}}");
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
                    Info("  {n}: {v}", kv.Key, kv.Value);
                }
            }
        }
    }

    [RequiresPreviewFeatures]
    static void PrintModelOps(string file)
    {
        ExitIfFileNotFound(file);
        var m = Model.Parse(file);
        if (m is null)
        {
            Exit(ExitResult.INVALID_INPUT);
            return;
        }
        Info("Graph details: Name: {name}. Domain: {dom}. Producer name: {pn}. Producer version: {pv}. IR Version: {ir}. DocString: {ds}.", m.Graph.Name, m.Domain, m.ProducerName, m.ProducerVersion, m.IrVersion.ToString(), m.Graph.DocString);
        Info("Graph has {count} input tensor(s): {in}", m.Graph.Input.Count, m.Graph.Input.Select(t => t.TensorNameDesc()));
        Info("Graph has {count} output tensor(s): {out}", m.Graph.Output.Count, m.Graph.Output.Select(t => t.TensorNameDesc()));
        Info("Graph has {count} initializer tensor(s): {out}", m.Graph.Initializer.Count, m.Graph.Initializer.Select(t => t.TensorNameDesc()));
        List<string> ops = new List<string>();
        foreach(var node in m.Graph.Node)
        {
            var op = Enum.Parse<OpType>(node.OpType);
            if (!ops.Contains(node.OpType))
            {
                ops.Add(node.OpType);
            }
        }
        Info("Printing list of distinct ONNX operations in model {f}...", file);
        foreach(var op in ops)
        {
            Con.Write(op + " ");
        }
        Con.Write(Environment.NewLine);
        Info("{d} total distinct operations in model.", m.Graph.Node.Count);
    }

    [RequiresPreviewFeatures]
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
        Info("Graph details: Name: {name}. Domain: {dom}. Producer name: {pn}. Producer version: {pv}. IR Version: {ir}. DocString: {ds}.", m.Graph.Name, m.Domain, m.ProducerName, m.ProducerVersion, m.IrVersion.ToString(), m.Graph.DocString);
        Info("Graph has {count} input tensors: {in}", m.Graph.Input.Count, m.Graph.Input.Select(t => t.TensorNameDesc()));
        Info("Graph has {count} output tensors: {out}", m.Graph.Output.Count, m.Graph.Output.Select(t => t.TensorNameDesc()));
        Info("Printing list of ONNX initializers in graph...");
        foreach (var i in initializers)
        {
            Con.WriteLine(i);
        }
        Info("{d} total initializers in model. * = initializer for graph input.", m.Graph.Initializer.Count);
    }

    [RequiresPreviewFeatures]
    static void Run(string file, IEnumerable<string> inputs)
    {
        ExitIfFileNotFound(file);
        var graph = Model.LoadFromFile(file);
        if (graph is null)
        {
            Exit(ExitResult.INVALID_INPUT);
            return;
        }
        Info("Graph details: Name: {name}. Domain: {dom}. Producer name: {pn}. Producer version: {pv}. IR Version: {ir}. DocString: {ds}.", graph.Metadata["Name"], graph.Metadata["Domain"], graph.Metadata["ProducerName"], graph.Metadata["ProducerVersion"], graph.Metadata["IrVersion"]?.ToString() ?? "", graph.Metadata["DocString"]);
        var ui = Data.GetInputTensorsFromFileArgs(inputs);
        if (ui is null)
        {
            Exit(ExitResult.INVALID_INPUT);
        }
        else
        {
            graph.Execute(ui);
        }
    }

    static void PrintLogo()
    {
        Con.Write(new FigletText(font, "Lokad.Onnx").Color(Color.Magenta1));
        Con.Write(new Text($"v{AssemblyVersion.ToString(3)}\n"));
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
        typeof(Options), typeof(InfoOptions), typeof(RunOptions)
        
    };
    static FigletFont font = FigletFont.Load(Path.Combine(AssemblyLocation, "chunky.flf"));
    static Dictionary<string, Type> optionTypesMap = new Dictionary<string, Type>();
    #endregion
}