namespace Lokad.Onnx.CLI;

using System;
using System.Globalization;
using System.Reflection;
using System.Runtime.Versioning;
using CommandLine;
using CommandLine.Text;
using Lokad.Onnx;
using Lokad.Onnx.Backend;

#region Enums
public enum ExitResult
{
    SUCCESS = 0,
    UNHANDLED_EXCEPTION = 1,
    INVALID_OPTIONS = 2,
    NOT_FOUND = 4,
    UNKNOWN_ERROR = 7
}
#endregion

[RequiresPreviewFeatures]
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
        Initialize("Lokad.Onnx.CLI", "CLI", (args.Contains("--debug") || args.Contains("-d")), true, true);
        PrintLogo();
        var result = new Parser().ParseArguments<Options, InfoOptions>(args);
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
                Error("A required option is missing: {0}.", error.NameInfo.NameText);
                Info(help);
                Exit(ExitResult.INVALID_OPTIONS);
            }
            else if (errors.Any(e => e.Tag == ErrorType.UnknownOptionError))
            {
                UnknownOptionError error = (UnknownOptionError)errors.First(e => e.Tag == ErrorType.UnknownOptionError);
                help.AddVerbs(optionTypes);
                Error("Unknown option: {error}.", error.Token);
                Info(help);
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
            else
            {
                PrintModelInfo(io.File);
            }
        });
    }
    #endregion

    static void PrintModelInfo(string file)
    {
        ExitIfFileNotFound(file);
        var graph = Model.LoadFromFile(file);
    }

    static void PrintModelOps(string file)
    {
        ExitIfFileNotFound(file);
        var m = Model.Parse(file);
        if (m is null)
        {
            Error("Could not parse file {f} as ONNX model.", file);
            Exit(ExitResult.UNKNOWN_ERROR);
            return;
        }
        List<string> ops = new List<string>();
        foreach(var node in m.Graph.Node)
        {
            var op = Enum.Parse<OpType>(node.OpType);
            if (!ops.Contains(node.OpType))
            {
                ops.Add(node.OpType); 
            }
        }
        Info("Printing list of ONNX operations in model {f}...", file);
        foreach(var op in ops)
        {
            Con.Write(op + " ");
        }
        Con.WriteLine("");
        Warn("{d} total operations in model.", m.Graph.Node.Count);
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
            return h;
        },
        e =>
        {
            return e;
        });
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
        typeof(Options), typeof(InfoOptions)
        
    };
    static FigletFont font = FigletFont.Load(Path.Combine(AssemblyLocation, "chunky.flf"));
    static Dictionary<string, Type> optionTypesMap = new Dictionary<string, Type>();
    #endregion
}