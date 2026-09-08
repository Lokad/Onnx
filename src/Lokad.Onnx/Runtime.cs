namespace Lokad.Onnx
{
    using System;
    using System.IO;
    using System.Net;
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
        public static void Error(Exception ex, string messageTemplate, params object?[] args) => Log.Write(LogLevel.Error, messageTemplate + " | " + ex.ToString(), args);

        [DebuggerStepThrough]
        public static void Warn(string messageTemplate, params object?[] args) => Log.Write(LogLevel.Warn, messageTemplate, args);

        [DebuggerStepThrough]
        public static void Fatal(string messageTemplate, params object?[] args) => Log.Write(LogLevel.Fatal, messageTemplate, args);

        [DebuggerStepThrough]
        public static LoggerOp Begin(string messageTemplate, params object?[] args) =>
            Log.Sink is null ? LoggerOp.Silent : new LoggerOp(messageTemplate, args);

        public static bool DownloadFile(string name, Uri downloadUrl, string downloadPath)
        {
#pragma warning disable SYSLIB0014 // Type or member is obsolete
            using (var op = Begin("Downloading {0} from {1} to {2}", name, downloadUrl, downloadPath))
            {
                if (File.Exists(downloadPath)) Warn("File {0} exists, overwriting...", downloadPath);
                using (var client = new WebClient())
                {
                    client.DownloadProgressChanged += (object sender, DownloadProgressChangedEventArgs e) =>
                    {
                        Info("Received {b} bytes from of {t} for {p}.", e.BytesReceived, e.TotalBytesToReceive, downloadPath);

                    };
                    client.DownloadDataCompleted += (object sender, DownloadDataCompletedEventArgs e) =>
                    {

                    };
                    client.DownloadFile(downloadUrl, downloadPath);
                }
                if (File.Exists(downloadPath))
                {
                    op.Complete();
                    return true;
                }
                else
                {
                    Error("Did not locate file at {p}.", downloadPath);
                    return false;
                }
            }
#pragma warning restore SYSLIB0014 // Type or member is obsolete
        }
    }
}
