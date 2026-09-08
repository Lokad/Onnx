namespace Lokad.Onnx;

using System;
using System.Diagnostics;

public class LoggerOp :  IDisposable
{
    static readonly LoggerOp silent = new LoggerOp(true);

    /// <summary>
    /// Shared no-op scope returned when no sink is installed: completing,
    /// abandoning or disposing it does nothing, so silent runs allocate
    /// nothing per scope. Never mutate it.
    /// </summary>
    internal static LoggerOp Silent => silent;

    private readonly bool isSilent;

    private LoggerOp(bool silent)
    {
        isSilent = silent;
    }

    public LoggerOp(string opName, params object?[] args)
    {
        timer.Start();
        this.opName = opName;
        Log.Write(LogLevel.Info, opName + "...", args);
    }

    public void Complete()
    {
        if (isSilent) return;
        timer.Stop();
        Log.Write(LogLevel.Info, "{0} completed in {1}ms.", opName, timer.ElapsedMilliseconds);
        isCompleted = true;
    }

    public void Abandon()
    {
        if (isSilent) return;
        timer.Stop();
        Log.Write(LogLevel.Error, "{0} abandoned after {1}ms.", opName, timer.ElapsedMilliseconds);
        isAbandoned = true;
    }

    public void Dispose()
    {
        if (isSilent) return;
        if (timer.IsRunning) timer.Stop();
        if (!(isCompleted || isAbandoned))
        {
            isAbandoned = true;
            Log.Write(LogLevel.Error, "{0} abandoned after {1}ms.", opName, timer.ElapsedMilliseconds);
        }
    }

    public string opName = "";

    public Stopwatch timer = new Stopwatch();

    protected bool isCompleted = false;

    protected bool isAbandoned = false;
}
