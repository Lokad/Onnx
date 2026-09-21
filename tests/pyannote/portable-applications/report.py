"""Report the closed public qualification without treating its times as a comparison."""
from common import *


def main():
    proof, analysis = read(BASE / 'closed.json'), read(BASE / 'analysis.json')
    assert proof['passed'] and analysis['passed'] and proof['analysis'] == pin(BASE / 'analysis.json')
    verify(proof['files'])
    for identity in proof['identities']:
        terminal(identity)
    report, observations = TOOLS / 'results-20260922.md', TOOLS / 'observations-20260922.json'
    assert not report.exists() and not observations.exists()
    save(observations, dict(closure=pin(BASE / 'closed.json'), build_closure=pin(COMPLETE / 'closed.json'),
        test_closure=pin(TESTS / 'closed.json'), **analysis))
    rows = '\n'.join(f"| {row['name']} | {row['seconds']:.3f} | {row['allocated_bytes']:,} |" for row in analysis['meetings'])
    report.write_text(f'''# Complete pyannote application qualification of the integrated build

The normal source/package runtime passes all **16 complete dialogue requests**,
both **ten-minute meetings** and the final **30-second recovery** request.
Every public result matches the accepted sparse-mel predecessor exactly.
Both ordinary and exclusive speaker timelines match the retained Microsoft ORT
references exactly; centroid comparisons pass the original unchanged bounds.
Inputs and held outputs remain unchanged across requests.

The runtime includes the reviewed convolution/LSTM work, request-scoped contexts,
pooled convolution outputs, portable row grouping, sparse mel frontend and the
LSTM storage guard. This checks the exact fresh binaries produced by the
[normal source/package build](../portable-integration/results-20260922.md).
The [self-contained tests](../portable-integration-tests/results-20260922.md)
separately pass3,290backend cases with93existing skips,342tensor cases and89frontend
cases in each normal/hardware-disabled mode. Those successful suites and the
package check are reused; only the complete public consumers run here.

Core: `{analysis['core']['sha256']}`.
Data: `{analysis['data']['sha256']}`.
The original AudioBenchmark and NaturalMeetings consumers are byte-identical
to their predecessors. No product, consumer or test assembly is rebuilt.
Their saved results identify the exact loaded product/consumer bytes.

| Request | Observed seconds | Cumulative managed allocated bytes |
|---|---:|---:|
{rows}

These times and allocations are descriptive. There is no fresh native timing
worker or matched performance comparison in this qualification. They do not
change the accepted Windows10.493s versus ORT6.320s result, which retains its
original Core5c0/Datae9 identities, or the older AMD production baselines.
The largest native meeting centroid error is
{max(row['maximum_centroid_error'] for row in analysis['meetings']):.12g}.

All **{analysis['resource_samples']} resource samples** pass and all **{len(analysis['identities'])} process identities**
are terminal. Peak sampled owned RSS is **{analysis['peak_rss']:,} bytes**.
Windows CPU2,normal.NET10.0.12,10GiB preflight,8GiB RSS ceiling,1GiB available,
20GiB disk and1GiB output limits apply. Dialogue has a900second ceiling and
meeting workers3600seconds. Every output and resource record is retained.
Controller session48724 exited0 before the independent audit.

This closes complete public behavior for the portable integrated build.
AMD selection, qualification of any different combined dispatch and actual root
product integration remain pending. Pyannote stays first, Parakeet second,
Whisper deferred. The live e5 campaign and queued primary AMD payload are unchanged.

Closure: `{pin(BASE / 'closed.json')['sha256']}` ({pin(BASE / 'closed.json')['bytes']:,} bytes).
[Full observations](observations-20260922.json) record every dialogue duration,
meeting/native comparison, allocation total and process/resource summary.
Run `run.py`, then `audit.py` only after controller exit, then this report tool;
all use `C:/Python313/python.exe -X utf8 -B` from the repository root.
Existing evidence directories are refused, never overwritten.
''', encoding='utf8')
    print(json.dumps(dict(report=rel(report), closed=pin(BASE / 'closed.json'))))


if __name__ == '__main__':
    main()
