namespace Lokad.Onnx;

using System;
using System.Diagnostics;

public class LoggerOp :  IDisposable
{
    public LoggerOp(string opName, params object?[] args)
    {
        timer.Start();
        this.opName = opName;
        Log.Write(LogLevel.Info, opName + "...", args);
    }

    public void Complete()
    {
        timer.Stop();
        Log.Write(LogLevel.Info, "{0} completed in {1}ms.", opName, timer.ElapsedMilliseconds);
        isCompleted = true;
    }

    public void Abandon()
    {
        timer.Stop();
        Log.Write(LogLevel.Error, "{0} abandoned after {1}ms.", opName, timer.ElapsedMilliseconds);
        isAbandoned = true;
    }

    public void Dispose()
    {
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
