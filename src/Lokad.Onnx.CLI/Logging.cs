namespace Lokad.Onnx.CLI
{
    public static class Logging
    {        
        [DebuggerStepThrough]
        public static void Info(Logger logger, string messageTemplate, params object[] args) => logger.Info(messageTemplate, args);

        [DebuggerStepThrough]
        public static void Debug(Logger logger, string messageTemplate, params object[] args) => logger.Debug(messageTemplate, args);

        [DebuggerStepThrough]
        public static void Error(Logger logger, string messageTemplate, params object[] args) => logger.Error(messageTemplate, args);

        [DebuggerStepThrough]
        public static void Error(Logger logger, Exception ex, string messageTemplate, params object[] args) => logger.Error(ex, messageTemplate, args);

        [DebuggerStepThrough]
        public static void Warn(Logger logger, string messageTemplate, params object[] args) => logger.Warn(messageTemplate, args);

        [DebuggerStepThrough]
        public static void Fatal(Logger logger, string messageTemplate, params object[] args) => logger.Fatal(messageTemplate, args);

        [DebuggerStepThrough]
        public static Logger.Op Begin(Logger logger, string messageTemplate, params object[] args) => logger.Begin(messageTemplate, args);
    }
}
