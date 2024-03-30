namespace Lokad.Onnx
{
    using System;
    using System.IO;
    using System.Linq;
    using System.Reflection;
    using System.Threading;

    public abstract class Runtime
    {
        #region Constructors
        static Runtime()
        {
            AppDomain.CurrentDomain.UnhandledException += AppDomain_UnhandledException;
            EntryAssembly = Assembly.GetEntryAssembly();
            IsUnitTestRun = EntryAssembly?.FullName.StartsWith("testhost") ?? false;
            SessionId = Rng.Next(0, 99999);            
            Logger = new ConsoleLogger2();
        }

        public Runtime(CancellationToken ct)
        {
            Ct = ct;
        }

        public Runtime() : this(Cts.Token) { }
        #endregion

        #region Properties
        public static bool RuntimeInitialized { get; protected set; }

        public static bool DebugEnabled { get; set; }

        public static bool InteractiveConsole { get; set; } = false;

        public static string PathSeparator { get; } = Environment.OSVersion.Platform == PlatformID.Win32NT ? "\\" : "/";

        public static string ToolName { get; set; } = "Lokad ONNX";
        
        public static string LogName { get; set; } = "BASE";

        public static Logger Logger { get; protected set; }

        public static string UserHomeDir => Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

        public static string AppDataDir => Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);

        public static string LokadDevDir => Path.Combine(AppDataDir, "LokadDev");

        public static Random Rng { get; } = new Random();

        public static int SessionId { get; protected set; }

        public static CancellationTokenSource Cts { get; } = new CancellationTokenSource();

        public static CancellationToken Ct { get; protected set; } = Cts.Token;

        public static Assembly? EntryAssembly { get; protected set; }

        public static string AssemblyLocation { get; } = Path.GetDirectoryName(Assembly.GetAssembly(typeof(Runtime))!.Location)!;

        public static Version AssemblyVersion { get; } = Assembly.GetAssembly(typeof(Runtime))!.GetName().Version!;
        
        public static bool IsUnitTestRun { get; set; }

        public static string RunFile => LokadDevDir.CombinePath(ToolName + ".run");
        #endregion

        #region Methods
        public static void Initialize(string toolname, string logname, bool debug = false, bool logToConsole = false, bool colorConsole=false)
        {
            lock (__lock)
            {
                Info("Initialize called on thread id {0}.", Thread.CurrentThread.ManagedThreadId);
                if (RuntimeInitialized)
                {
                    Info("Runtime already initialized.");
                    return;
                }
                if (!IsUnitTestRun)
                {
                    Logger.Close();
                    ToolName = toolname;
                    LogName = logname;
                    var fulllogfilename = LokadDevDir.CombinePath($"{ToolName}.{SessionId}.log");
                    Logger = new FileLogger(fulllogfilename, debug, LogName, logToConsole, colorConsole);
                    Info("{0} initialized from entry assembly {1} with log file {2}...", ToolName, EntryAssembly?.GetName().FullName ?? "(none)", fulllogfilename);
                    if (debug)
                    {
                        DebugEnabled = true;
                        Info("Debug mode enabled.");
                    }
                }
                RuntimeInitialized = true;
            }
        }
       
        [DebuggerStepThrough]
        public static void Info(string messageTemplate, params object[] args) => Logger.Info(messageTemplate, args);

        [DebuggerStepThrough]
        public static void Debug(string messageTemplate, params object[] args) => Logger.Debug(messageTemplate, args);

        [DebuggerStepThrough]
        public static void Error(string messageTemplate, params object[] args) => Logger.Error(messageTemplate, args);

        [DebuggerStepThrough]
        public static void Error(Exception ex, string messageTemplate, params object[] args) => Logger.Error(ex, messageTemplate, args);

        [DebuggerStepThrough]
        public static void Warn(string messageTemplate, params object[] args) => Logger.Warn(messageTemplate, args);

        [DebuggerStepThrough]
        public static void Fatal(string messageTemplate, params object[] args) => Logger.Fatal(messageTemplate, args);

        [DebuggerStepThrough]
        public static Logger.Op Begin(string messageTemplate, params object[] args) => Logger.Begin(messageTemplate, args);

        [DebuggerStepThrough]
        public static string FailIfFileNotFound(string filePath)
        {
            if (filePath.StartsWith("http://") || filePath.StartsWith("https://"))
            {
                return filePath;
            }
            else if (!File.Exists(filePath))
            {
                throw new FileNotFoundException(filePath);
            }
            else return filePath;
        }

        [DebuggerStepThrough]
        public static string WarnIfFileExists(string filename)
        {
            if (File.Exists(filename)) Warn("File {0} exists, overwriting...", filename);
            return filename;
        }


        [DebuggerStepThrough]
        public static object? GetProp(object o, string name)
        {
            PropertyInfo[] properties = o.GetType().GetProperties(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            return properties.FirstOrDefault(x => x.Name == name)?.GetValue(o);
        }

        private static void AppDomain_UnhandledException(object sender, UnhandledExceptionEventArgs e)
        {
            if (Logger != null)
            {
                Error((Exception)e.ExceptionObject, "Unhandled runtime error occurred.");
            }
        }

        public static string? RunCmd(string cmdName, string arguments = "", string? workingDir = null, DataReceivedEventHandler? outputHandler = null, DataReceivedEventHandler? errorHandler = null,
            bool checkExists = true, bool isNETFxTool = false, bool isNETCoreTool = false)
        {
            if (checkExists && !(File.Exists(cmdName) || (isNETCoreTool && File.Exists(cmdName.Replace(".exe", "")))))
            {
                Error("The executable {0} does not exist.", cmdName);
                return null;
            }
            using (Process p = new Process())
            {
                var output = new StringBuilder();
                var error = new StringBuilder();
                p.StartInfo.UseShellExecute = false;
                p.StartInfo.RedirectStandardInput = false;
                p.StartInfo.RedirectStandardOutput = true;
                p.StartInfo.RedirectStandardError = true;
                p.StartInfo.CreateNoWindow = true;

                if (isNETFxTool && System.Environment.OSVersion.Platform == PlatformID.Unix)
                {
                    p.StartInfo.FileName = "mono";
                    p.StartInfo.Arguments = cmdName + " " + arguments;
                }
                else if (isNETCoreTool && System.Environment.OSVersion.Platform == PlatformID.Unix)
                {
                    p.StartInfo.FileName = File.Exists(cmdName) ? cmdName : cmdName.Replace(".exe", "");
                    p.StartInfo.Arguments = arguments;

                }
                else
                {
                    p.StartInfo.FileName = cmdName;
                    p.StartInfo.Arguments = arguments;
                }

                p.OutputDataReceived += (sender, e) =>
                {
                    if (e.Data is not null)
                    {
                        output.AppendLine(e.Data);
                        Debug(e.Data);
                        outputHandler?.Invoke(sender, e);
                    }
                };
                p.ErrorDataReceived += (sender, e) =>
                {
                    if (e.Data is not null)
                    {
                        error.AppendLine(e.Data);
                        Error(e.Data);
                        errorHandler?.Invoke(sender, e);
                    }
                };
                if (workingDir is not null)
                {
                    p.StartInfo.WorkingDirectory = workingDir;
                }
                Debug("Executing cmd {0} in working directory {1}.", cmdName + " " + arguments, p.StartInfo.WorkingDirectory);
                try
                {
                    p.Start();
                    p.BeginOutputReadLine();
                    p.BeginErrorReadLine();
                    p.WaitForExit();
                    return error.ToString().IsNotEmpty() ? null : output.ToString();
                }

                catch (Exception ex)
                {
                    Error(ex, "Error executing command {0} {1}", cmdName, arguments);
                    return null;
                }
            }
        }

        public static void CopyDirectory(string sourceDir, string destinationDir, bool recursive = false)
        {
            using var op = Begin("Copying {0} to {1}", sourceDir, destinationDir);
            // Get information about the source directory
            var dir = new DirectoryInfo(sourceDir);

            // Check if the source directory exists
            if (!dir.Exists)
                throw new DirectoryNotFoundException($"Source directory not found: {dir.FullName}");

            // Cache directories before we start copying
            DirectoryInfo[] dirs = dir.GetDirectories();

            // Create the destination directory
            Directory.CreateDirectory(destinationDir);

            // Get the files in the source directory and copy to the destination directory
            foreach (FileInfo file in dir.GetFiles())
            {
                string targetFilePath = Path.Combine(destinationDir, file.Name);
                file.CopyTo(targetFilePath);
            }

            // If recursive and copying subdirectories, recursively call this method
            if (recursive)
            {
                foreach (DirectoryInfo subDir in dirs)
                {
                    string newDestinationDir = Path.Combine(destinationDir, subDir.Name);
                    CopyDirectory(subDir.FullName, newDestinationDir, true);
                }
            }
            op.Complete();
        }

        public static string ViewFilePath(string path, string? relativeTo = null)
        {
            if (!DebugEnabled)
            {
                if (path is null)
                {
                    return string.Empty;
                }
                else if (relativeTo is null)
                {
                    return (Path.GetFileName(path) ?? path);
                }
                else
                {
                    return (IOExtensions.GetRelativePath(relativeTo, path));
                }
            }
            else return path;
        }
        #endregion

        #region Fields
        protected static object __lock = new object();
        #endregion
    }
}