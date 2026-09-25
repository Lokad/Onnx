"""Publish the successful clock proof alongside both retained instrument failures."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent


def read(path): return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def main():
    reports={}
    for version,passed in [('',False),('-v2',False),('-v3',True)]:
        folder=ROOT/f'artifacts/e5-frequency-proof{version}-amd-20260925'
        proof=read(folder/'closed.json')
        assert proof['passed']==passed and proof['analysis']==pin(folder/'analysis.json')
        for name,wanted in proof['files'].items(): assert pin(folder/name)==wanted,name
        reports[version or 'original']=dict(closure=pin(folder/'closed.json'),analysis=read(folder/'analysis.json'))
    final=reports['-v3']['analysis']
    assert final['passed'] and final['epoch']['uncertainty_ns']<=20_000_000
    assert final['timestamp_brackets']==1000 and final['inference_calls']==0
    assert final['whole_workload_intervals'] and final['samples']==75
    base=ROOT/'artifacts/e5-frequency-proof-v3-amd-20260925'
    document=OUT/'clock-proof-20260925.md'; data=OUT/'clock-proof-20260925.json'; counters=OUT/'proof-intervals-20260925.csv'
    assert not any(p.exists() for p in [document,data,counters])
    data.write_text(json.dumps(dict(reports=reports,source=pin(Path(__file__))),indent=2,allow_nan=False)+'\n',encoding='utf8')
    counters.write_bytes((base/'collected/counters/intervals.csv').read_bytes())
    document.write_text(f'''# Frequency-counter clock alignment proof

**The corrected no-model proof passes.** Its perf epoch is bounded to
{final['epoch']['uncertainty_ns']/1e6:.6f} ms, below the prospective 20 ms limit.
All 1,000 calls to the installed .NET timestamp function lie inside their
Linux monotonic-clock brackets. A complete counter interval lies within the
known arithmetic workload and reports positive APERF, MPERF and TSC counts.
All three raw intervals remain retained. No model or product build ran.

The first attempt rejected an incorrect forced-output assumption. Although
enable is acknowledged, the handler returns zero and the caller's positive-result
branch does not print an immediate interval. The proof refused to treat the
next periodic reading as a 20 ms handshake. The second attempt completed the
arithmetic workload but rejected the next acknowledgment: the reader consumed
four bytes while perf sends five, including a NUL terminator. Both failed runs
and all their source/data remain retained. See the
[control handler](https://raw.githubusercontent.com/torvalds/linux/v6.17/tools/perf/util/evlist.c)
and [interval dispatch](https://raw.githubusercontent.com/torvalds/linux/v6.17/tools/perf/builtin-stat.c).

The successful component reads the exact acknowledgment. It bounds the initial
epoch between launch and enable acknowledgment. At exit, ping establishes that
earlier interval output is complete; the proof snapshots it, stops the bounded
helper and requires new interval output before perf exits. Subtracting that
reading's relative timestamp gives a second epoch bound. The two bounds overlap.
The final handshake and shared uncertainty both remain within 20 ms.

Ten local checks pass, including two successive fragmented acknowledgments,
missing/duplicate/malformed counters and incompatible epoch bounds. The actual
proof passes {final['samples']} resource observations with peak owned RSS
{final['peak_rss']:,} bytes. All supervisor, perf and workload owners are terminal.
No system settings, warmup, scoring limit or release product changed.

This proves the observation mechanism, not that frequency explains e5 timings.
The next workload is two identical original M73 candidate e5-512tok processes,
each preserving all 780 calls and the original numerical/ownership assertions.
Counter intervals and events must all be retained; no diagnostic clock replaces
the original failed release comparison.

[Closures, bounds and retained failures](clock-proof-20260925.json),
[every proof interval](proof-intervals-20260925.csv).

Successful closure: `{reports['-v3']['closure']['sha256']}`.
''',encoding='utf8')
    print(json.dumps(dict(passed=True,closure=reports['-v3']['closure'],uncertainty_ms=final['epoch']['uncertainty_ns']/1e6,inference_calls=0)))


if __name__=='__main__':main()
