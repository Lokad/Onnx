"""Add external counter brackets to the frozen current-root Pad workload."""
import importlib.util
from pathlib import Path
from protocol import pin, read

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-pad-memory-diagnostic-amd-20260926'
PRIOR = ROOT/'tests/parakeet/pad-current-screen'
loader = importlib.util.spec_from_file_location('original_screen_prepare', PRIOR/'prepare.py')
parent = importlib.util.module_from_spec(loader); loader.loader.exec_module(parent)
original_consumer = parent.consumer
original_previous = parent.previous_closed


def previous_closed():
    for name,wanted in read(TOOLS/'dependencies.json').items():assert pin(ROOT/name)==wanted,name
    original_previous()
    folder=ROOT/'artifacts/parakeet-pad-current-screen-amd-20260926'
    assert pin(folder/'closed.json')['sha256']=='5b8d7df749d600238dfc6e1c9ead50486b0027754656c57d0c553573ada1b17d'
    value=read(folder/'closed.json');assert value['passed'] and not value['admitted']
    for name,wanted in value['files'].items():assert pin(folder/name)==wanted,name
    report=ROOT/'tests/parakeet/pad-memory-results/ort-allocation-observations-20260926.json'
    value=read(report);assert value['inference_calls']==0 and value['unique_pattern_keys']==20
    for name,wanted in value['inputs'].items():assert pin(ROOT/name)==wanted,name
    for row in value['groups']:
        if row['measured']:assert row['zero_requested_delta']==row['calls']==1440


def replacements():
    # Every replacement is independently reversible, including the suffix timer.
    return [
        ('                long start = Stopwatch.GetTimestamp();',
         '                var beforeCounters = MemoryCounters.Before();\n                long start = Stopwatch.GetTimestamp();'),
        ('                long ticks = Stopwatch.GetTimestamp() - start;',
         '                long stop = Stopwatch.GetTimestamp();\n                long ticks = stop - start;'),
        ('                Require(ticks > 0 && returned.Status == OpStatus.Success, "clock/status");',
         '                var sample = MemoryCounters.After(beforeCounters, start, stop);\n                Require(ticks > 0 && returned.Status == OpStatus.Success, "clock/status");'),
        ('clocks.Add(new { iteration, warmup = iteration < 600, start, stop, ticks });',
         'clocks.Add(new { iteration, warmup = iteration < 600, start, stop, ticks, counters = sample });'),
        ('clocks.Add(new { iteration, warmup = iteration < 600, ticks });',
         'clocks.Add(new { iteration, warmup = iteration < 600, ticks, counters = sample });'),
        ('        long suffixStart = Prime(',
         '        MemoryCounters.Calibrate(Path.GetDirectoryName(Path.GetFullPath(args[3]))!, "before");\n        long suffixStart = Prime('),
        ('        using var output = new FileStream(args[3], FileMode.CreateNew);',
         '        MemoryCounters.Calibrate(Path.GetDirectoryName(Path.GetFullPath(args[3]))!, "after");\n        using var output = new FileStream(args[3], FileMode.CreateNew);'),
        ('protocol = "parakeet-pad-public-600-180-v1", role, sequence,',
         'protocol = "parakeet-pad-public-600-180-v1", diagnosticOnly = true, nativeThread = MemoryCounters.gettid(), role, sequence,')]


def consumer():
    original=original_consumer(); result=original
    for before,after in replacements():
        assert before in result
        result=result.replace(before,after)
    restored=result
    # The inserted suffix stop has the same text as the original prefix stop;
    # reverse the suffix replacement only after splitting off Main.
    for before,after in reversed(replacements()):
        if before=='                long ticks = Stopwatch.GetTimestamp() - start;':
            at=restored.index('    static void Main(string[] args)')
            restored=restored[:at]+restored[at:].replace(after,before)
        else:restored=restored.replace(after,before)
    assert restored==original, 'Only the reviewed external counter brackets may change'
    return result+'\n'+(TOOLS/'Counters.cs').read_text(encoding='utf8')


parent.BASE=BASE;parent.TOOLS=TOOLS
parent.previous_closed=previous_closed;parent.consumer=consumer
prepare=parent.prepare

if __name__=='__main__':prepare()
