namespace Lokad.Onnx.Interop;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

using NLog;
using NLog.Config;

public class Graph : Runtime
{
    public static ComputationalGraph? Load(string filepath) => Model.Load(filepath);

    public static ComputationalGraph? Load(byte[] buffer) => Model.Load(buffer);

    public static ITensor[]? GetInputTensorsFromFileArgs(IEnumerable<string> args, bool saveInput = false) => Data.GetInputTensorsFromFileArgs(args, saveInput);

    public static ITensor? GetInputTensorFromFileArg(string arg, bool saveInput = false) => Data.GetInputTensorsFromFileArgs(new[] { arg }, saveInput)?.First();

    public static ITensor[]? GetTextTensors(string text, string props) => Text.GetTextTensors(text, props);

    public static ITensor[]? GetTextTensors(string[] text, string props) => Text.GetTextTensors(text, props);

    public static void SetDebugMode() => Runtime.Initialize("ONNX", "PYTHON", true, ConfigureLogger(true));

    protected static Microsoft.Extensions.Logging.ILogger ConfigureLogger(bool debug)
    {
        var config = new LoggingConfiguration();
        if (debug)
        {
            config.Variables["logLevel"] = "Debug";
        }
        var logconsole = new NLog.Targets.ColoredConsoleTarget("logconsole");
        config.AddTarget(logconsole);
        config.AddRule(new LoggingRule("*", LogLevel.Info, logconsole));
        config.AddRule(new LoggingRule("*", LogLevel.Warn, logconsole));
        config.AddRule(new LoggingRule("*", LogLevel.Error, logconsole));
        config.AddRule(new LoggingRule("*", LogLevel.Fatal, logconsole));
        if (debug)
        {
            config.AddRule(new LoggingRule("*", LogLevel.Debug, logconsole));
        }
        LogManager.Configuration = config;  
        return new NLog.Extensions.Logging.NLogLoggerFactory(
            new NLog.Extensions.Logging.NLogProviderOptions()
            {
                AutoShutdown = true,
                ParseMessageTemplates = true,
                CaptureMessageTemplates = true,
            }).CreateLogger("console");
    }
}

