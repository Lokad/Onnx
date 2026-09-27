"""Add bounded diagnostic markers to the frozen complete-call consumer only."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCREEN = ROOT/'artifacts/parakeet-decoder-lstm-layout-timing-amd-20260927'


def instrument():
    original = (SCREEN/'bundle/source/Timing.cs').read_text()
    text = original.replace('using System.Diagnostics;', 'using System.Diagnostics;\nusing System.Diagnostics.Tracing;')
    before = '    static int Main(string[] args)'
    text = text.replace(before, '''    [DllImport("libc")] static extern int gettid();
    readonly record struct DiagnosticRow(int ordinal, string phase, int repeat, int call,
        long marker, long start, long stop, long afterMarker,
        int gc0, int gc1, int gc2, int after0, int after1, int after2,
        long pauseBefore, long pauseAfter);

'''+before)
    before = '        using var specDoc = JsonDocument.Parse'
    insert = '''        Require(role == "selectedfallback" && mode == "512", "One unchanged normal fallback observation");
        int nativeThread = gettid(); var events = MatrixEvents.Log;
        var diagnostics = new DiagnosticRow[3800]; int ordinal = 0;
        string traceDirectory = Path.GetDirectoryName(Path.GetFullPath(args[4]))!;
        File.WriteAllText(Path.Combine(traceDirectory, "ready.json"), JsonSerializer.Serialize(new {
            pid = Environment.ProcessId, native_thread = nativeThread, counter = Stopwatch.GetTimestamp() }));
        var waiting = Stopwatch.StartNew();
        while (!events.IsEnabled(EventLevel.Informational, (EventKeywords)1))
        {
            Require(waiting.Elapsed.TotalSeconds < 30, "Collector did not enable markers"); Thread.Sleep(10);
        }
        File.WriteAllText(Path.Combine(traceDirectory, "collector-enabled.json"), JsonSerializer.Serialize(new {
            pid = Environment.ProcessId, counter = Stopwatch.GetTimestamp() }));
'''
    assert text.count(before) == 1; text = text.replace(before, insert+before)
    before = '                var context = contexts[call.index];long allocation = GC.GetAllocatedBytesForCurrentThread(), begin = Stopwatch.GetTimestamp();'
    after = '''                var context = contexts[call.index];
                int gc0 = GC.CollectionCount(0), gc1 = GC.CollectionCount(1), gc2 = GC.CollectionCount(2);
                long pauseBefore = GC.GetTotalPauseDuration().Ticks;
                long marker = Stopwatch.GetTimestamp(); events.Begin(0, ordinal, marker);
                long allocation = GC.GetAllocatedBytesForCurrentThread(), begin = Stopwatch.GetTimestamp();'''
    assert text.count(before) == 1; text = text.replace(before, after)
    before = '                long ticks = Stopwatch.GetTimestamp() - begin, allocated = GC.GetAllocatedBytesForCurrentThread() - allocation;'
    after = '''                long stop = Stopwatch.GetTimestamp(), ticks = stop - begin, allocated = GC.GetAllocatedBytesForCurrentThread() - allocation;
                events.End(0, ordinal, stop); long afterMarker = Stopwatch.GetTimestamp();
                diagnostics[ordinal] = new(ordinal, phase, repeat, ordinal % 380, marker, begin, stop, afterMarker,
                    gc0, gc1, gc2, GC.CollectionCount(0), GC.CollectionCount(1), GC.CollectionCount(2),
                    pauseBefore, GC.GetTotalPauseDuration().Ticks);
                ordinal++;'''
    assert text.count(before) == 1; text = text.replace(before, after)
    before = '        Require(rows.Count == 3800 && held.Count > 0, "All warm and measured clocks retained");'
    assert text.count(before) == 1; text = text.replace(before, before+'\n        Require(ordinal == 3800 && gettid() == nativeThread, "Every diagnostic interval on original worker thread");')
    before = 'JsonSerializer.Serialize(output, new { passed = true, role, mode, worker, pid = Environment.ProcessId'
    after = 'JsonSerializer.Serialize(output, new { passed = true, diagnostic_only = true, native_thread = nativeThread, diagnostics, role, mode, worker, pid = Environment.ProcessId'
    assert text.count(before) == 1; text = text.replace(before, after)
    marker = (ROOT/'tests/parakeet/decoder-projection-observation/Driver.cs').read_text()
    marker = marker[marker.index('[EventSource(Name = "Lokad-Parakeet-MatMul-Diagnostic")]'):]
    text += '\n'+marker
    project = (SCREEN/'bundle/source/Timing.csproj').read_text()
    assert project.count('../products/candidate/') == 3
    project = project.replace('../products/candidate/', '../runtime/').replace('<Nullable>enable</Nullable>', '<Nullable>enable</Nullable><AllowUnsafeBlocks>true</AllowUnsafeBlocks>')
    return text, project
