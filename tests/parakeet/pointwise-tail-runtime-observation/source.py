"""Instrument the frozen raw-kernel timer without changing its timed statements."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCREEN = ROOT/'artifacts/parakeet-pointwise-tail-timing-amd-20260927'


def instrument():
    text = (SCREEN/'bundle/contract-source/Program.cs').read_text()
    def replace(before, after):
        nonlocal text
        assert text.count(before) == 1, before
        text = text.replace(before, after)
    replace('using System.Diagnostics;', 'using System.Diagnostics;\nusing System.Diagnostics.Tracing;')
    replace('    static (long ticks,long allocated) Measure(Kernel kernel,Case c)',
        '    static (long ticks,long allocated) Measure(Kernel kernel,Case c,MatrixEvents events,int ordinal,string phase,int repeat,out DiagnosticRow diagnostic)')
    replace('            long allocated=GC.GetAllocatedBytesForCurrentThread();', '''            int gc0=GC.CollectionCount(0),gc1=GC.CollectionCount(1),gc2=GC.CollectionCount(2);
            long pauseBefore=GC.GetTotalPauseDuration().Ticks;
            long marker=Stopwatch.GetTimestamp();events.Begin(0,ordinal,marker);
            long allocated=GC.GetAllocatedBytesForCurrentThread();''')
    replace('            return(end-begin,GC.GetAllocatedBytesForCurrentThread()-allocated);', '''            long allocation=GC.GetAllocatedBytesForCurrentThread()-allocated;
            events.End(0,ordinal,end);long afterMarker=Stopwatch.GetTimestamp();
            diagnostic=new(ordinal,phase,repeat,c.Index,marker,begin,end,afterMarker,
                gc0,gc1,gc2,GC.CollectionCount(0),GC.CollectionCount(1),GC.CollectionCount(2),
                pauseBefore,GC.GetTotalPauseDuration().Ticks);
            return(end-begin,allocation);''')
    replace('    static int Main(string[] args)', '''    [DllImport("libc")] static extern int gettid();
    readonly record struct DiagnosticRow(int ordinal,string phase,int repeat,int call,
        long marker,long start,long stop,long afterMarker,
        int gc0,int gc1,int gc2,int after0,int after1,int after2,long pauseBefore,long pauseAfter);

    static int Main(string[] args)''')
    replace('        var context=new AssemblyLoadContext', '''        Require(role=="candidate","One unchanged candidate observation");
        int nativeThread=gettid();var events=MatrixEvents.Log;
        var diagnostics=new DiagnosticRow[400];int ordinal=0;
        string traceDirectory=Path.GetDirectoryName(output)!;
        File.WriteAllText(Path.Combine(traceDirectory,"ready.json"),JsonSerializer.Serialize(new {
            pid=Environment.ProcessId,native_thread=nativeThread,counter=Stopwatch.GetTimestamp()}));
        var waiting=Stopwatch.StartNew();
        while(!events.IsEnabled(EventLevel.Informational,(EventKeywords)1))
        {Require(waiting.Elapsed.TotalSeconds<30,"Collector did not enable markers");Thread.Sleep(10);}
        File.WriteAllText(Path.Combine(traceDirectory,"collector-enabled.json"),JsonSerializer.Serialize(new {
            pid=Environment.ProcessId,counter=Stopwatch.GetTimestamp()}));
        var context=new AssemblyLoadContext''')
    replace('                var t=Measure(kernel,c);Require(t.ticks>0', '''                var t=Measure(kernel,c,events,ordinal,pass<warmups?"warmup":"measured",pass<warmups?pass:pass-warmups,out diagnostics[ordinal]);
                ordinal++;Require(t.ticks>0''')
    replace('        using(var stream=new FileStream', '''        Require(ordinal==400 && gettid()==nativeThread,"Every interval on original thread");
        using(var stream=new FileStream''')
    replace('new{completed=true,passed=true,role,pid=',
        'new{completed=true,passed=true,diagnostic_only=true,native_thread=nativeThread,diagnostics,spec_sha256=FileHash(args[0]),role,pid=')
    markers = (ROOT/'tests/parakeet/decoder-projection-observation/Driver.cs').read_text()
    text += '\n'+markers[markers.index('[EventSource(Name = "Lokad-Parakeet-MatMul-Diagnostic")]'):]
    project = (SCREEN/'bundle/contract-source/TailContracts.csproj').read_text()
    project = project.replace('$(FrozenProductDirectory)', '../runtime')
    project = project.replace('<OutputType>Exe</OutputType>', '<OutputType>Exe</OutputType><AssemblyName>PointwiseTailRuntime</AssemblyName>')
    return text, project
