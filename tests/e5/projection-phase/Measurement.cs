using System.Diagnostics;
using System.Diagnostics.Tracing;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

[EventSource(Name="Lokad-Packed-Phase")]
sealed class Markers : EventSource
{
    public static readonly Markers Log=new();
    [Event(1,Level=EventLevel.Informational)]
    public unsafe void Boundary(int id,int edge,long ticks)
    {
        if(!IsEnabled())return;
        EventData* fields=stackalloc EventData[3];
        fields[0].DataPointer=(IntPtr)(&id);fields[0].Size=sizeof(int);
        fields[1].DataPointer=(IntPtr)(&edge);fields[1].Size=sizeof(int);
        fields[2].DataPointer=(IntPtr)(&ticks);fields[2].Size=sizeof(long);
        WriteEventCore(1,3,fields);
    }
}

static class Measurement
{
    public static readonly List<Interval> Rows=new();
    static double sink;
    static int nextId=-1;
    public static void Gate(string gate)
    {
        _=Markers.Log;
        Console.WriteLine("Waiting for gate "+Environment.ProcessId);
        var deadline=Stopwatch.StartNew();
        while(!File.Exists(gate)){if(deadline.Elapsed.TotalSeconds>90)throw new TimeoutException("Trace gate");Thread.Sleep(10);}
        // Initialize diagnostic paths before the old correctness/workload schedule.
        var row=Begin("diagnostic-initialize",1,1,1,0,0,"warmup",1);End(row,Stopwatch.GetTimestamp());Rows.Clear();
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    public static void Smoke(int value){for(int i=0;i<16;i++)sink+=Math.Sqrt(value+i);}

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static Interval Begin(string region,int m,int n,int k,int mode,int sample,string phase,int iterations)
    {
        var row=new Interval{Id=nextId++,Region=region,M=m,N=n,K=k,Mode=mode,Sample=sample,Phase=phase,Iterations=iterations};
        row.MarkerBeginBefore=Stopwatch.GetTimestamp();
        Markers.Log.Boundary(row.Id,0,row.MarkerBeginBefore);
        row.MarkerBeginAfter=Stopwatch.GetTimestamp();
        row.ProcessCpuBefore=Cpu(2);row.ThreadCpuBefore=Cpu(3);
        row.Start=Stopwatch.GetTimestamp();return row;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static double End(Interval row,long end)
    {
        row.End=end;row.ThreadCpuAfter=Cpu(3);row.ProcessCpuAfter=Cpu(2);
        row.MarkerEndBefore=Stopwatch.GetTimestamp();Markers.Log.Boundary(row.Id,1,row.MarkerEndBefore);row.MarkerEndAfter=Stopwatch.GetTimestamp();
        Rows.Add(row);return (row.End-row.Start)*1000.0/Stopwatch.Frequency;
    }

    static long Cpu(int clock)
    {
        if(OperatingSystem.IsLinux())
        {
            if(clock_gettime(clock,out var value)!=0)throw new InvalidOperationException("clock_gettime");
            return checked(value.Seconds*1000000000+value.Nanoseconds);
        }
        using var process=Process.GetCurrentProcess();return process.TotalProcessorTime.Ticks*100;
    }
    [DllImport("libc",SetLastError=true)]static extern int clock_gettime(int clock,out Timespec value);
    [StructLayout(LayoutKind.Sequential)]struct Timespec{public long Seconds,Nanoseconds;}
}

sealed class Interval
{
    public int Id{get;set;}public string Region{get;set;}="";public int M{get;set;}public int N{get;set;}public int K{get;set;}
    public int Mode{get;set;}public int Sample{get;set;}public string Phase{get;set;}="";public int Iterations{get;set;}
    public long MarkerBeginBefore{get;set;}public long MarkerBeginAfter{get;set;}public long Start{get;set;}public long End{get;set;}
    public long MarkerEndBefore{get;set;}public long MarkerEndAfter{get;set;}
    public long ProcessCpuBefore{get;set;}public long ProcessCpuAfter{get;set;}public long ThreadCpuBefore{get;set;}public long ThreadCpuAfter{get;set;}
}
