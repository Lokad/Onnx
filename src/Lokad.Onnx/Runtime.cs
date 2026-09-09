namespace Lokad.Onnx
{
    using System;
    using System.IO;
    using System.Reflection;

    /// <summary>
    /// Small process-independent utilities shared by the library and its hosts.
    /// Logging shorthands delegate to the in-core <see cref="Log"/> sink, which is
    /// null by default so the library runs silent; hosts such as the CLI assign it.
    /// This is a static utility class: library concepts must not inherit it, and
    /// touching it installs no process-wide handlers or shared cancellation state.
    /// </summary>
    public static class Runtime
    {
        static System.Reflection.Assembly ThisAssembly => Assembly.GetAssembly(typeof(Runtime)) ?? throw new InvalidOperationException("Cannot locate the Lokad.Onnx assembly.");

        public static string AssemblyLocation => Path.GetDirectoryName(ThisAssembly.Location) ?? throw new InvalidOperationException("Cannot locate the Lokad.Onnx assembly directory.");

        public static Version AssemblyVersion => ThisAssembly.GetName().Version ?? throw new InvalidOperationException("Cannot read the Lokad.Onnx assembly version.");

        [DebuggerStepThrough]
        public static void Info(string messageTemplate, params object?[] args) => Log.Write(LogLevel.Info, messageTemplate, args);

        [DebuggerStepThrough]
        public static void Debug(string messageTemplate, params object?[] args) => Log.Write(LogLevel.Debug, messageTemplate, args);

        [DebuggerStepThrough]
        public static void Error(string messageTemplate, params object?[] args) => Log.Write(LogLevel.Error, messageTemplate, args);

        [DebuggerStepThrough]
        public static void Error(Exception ex, string messageTemplate, params object?[] args)
        {
            if (!Log.IsEnabled(LogLevel.Error)) return;
            Log.Write(LogLevel.Error, messageTemplate + " | " + ex.ToString(), args);
        }

        [DebuggerStepThrough]
        public static void Warn(string messageTemplate, params object?[] args) => Log.Write(LogLevel.Warn, messageTemplate, args);

        [DebuggerStepThrough]
        public static void Fatal(string messageTemplate, params object?[] args) => Log.Write(LogLevel.Fatal, messageTemplate, args);

        [DebuggerStepThrough]
        public static LoggerOp Begin(string messageTemplate, params object?[] args) =>
            Log.Sink is null ? LoggerOp.Silent : new LoggerOp(messageTemplate, args);

    /// <summary>
    /// True for runtime failures no caller can handle (out of memory, stack
    /// exhaustion): these propagate instead of converting into ordinary
    /// operation or parse failures.
    /// </summary>
    public static bool IsFatal(Exception ex) => ex is OutOfMemoryException or StackOverflowException;
    }
}
