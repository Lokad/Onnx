"""Render all retained AMD GELU cases without rerunning any measurement."""
from pathlib import Path
import argparse
import json
from prepare import pin, read


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    base, output = args.artifact, args.output
    closed = read(base / 'closed.json')
    assert closed['closed'] and closed['execution_passed'] and closed['all_owned_processes_terminal']
    assert {p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file()} == set(closed['files']) | {'closed.json'}
    for name, wanted in closed['files'].items():
        assert pin(base / name) == wanted, name
    audit = read(base / 'audit.json')
    assert audit['verdict'] == closed['verdict']
    assert audit['execution_passed'] and audit['controls_passed'] and not audit['overall_passed']
    targets = [output / ('results-20260920.md'), output / ('observations-20260920.json')]
    assert all(not p.exists() for p in targets)
    bundle = read(base / 'payload/bundle.json')
    names = ['Product', 'CopyA', 'CopyB', 'Conditional']
    text = '''# Conditional GELU on AMD — 2026-09-20

The exact shortcut **fails its fixed kernel screen**. Duplicate controls pass,
but candidate improvement against the actual production kernel is only 1.22%
at padded-128 and 1.60% at 128 tokens, below the required 2%. Individual workers
also exceed the allowed 2% regression: padded-128 visit 1 is 11.44% slower,
128 visit 3 is 4.94% slower, and 512 visit 3 is 3.63% slower. Preserve these
observations and the failed verdict. This experiment changes no product code,
default, complete-model score or Microsoft ORT timing.

AMD EPYC 9V74, four vCPUs, .NET 10.0.8, SDK 10.0.204. Workers inherit CPU 2
before CLR startup; the supervisor uses CPU 0. Timing uses normal runtime
settings with no experimental overrides. Each call runs all twelve captured
e5 BiasGelu layers, including loop/delegate dispatch and conditional overhead.
Inputs, outputs and biases are allocated before timing; validation is outside.
These are repeated activation banks, not complete graph execution.

## Every case

Means are milliseconds per complete twelve-layer bank, using every measured
batch. Product calls the actual qualified assembly; CopyA and CopyB are two
identical copies of its arithmetic. Conditional skips the unused large erf
polynomial only when all eight lanes select the small polynomial.

| Tokens | Product ms | CopyA ms | CopyB ms | Conditional ms | Conditional / Product | CopyB / CopyA |
|---|---:|---:|---:|---:|---:|---:|
'''
    for c in audit['cases']:
        m = c['means_ms']
        text += '| ' + c['name'] + ' | ' + ' | '.join(f'{m[n]:.6f}' for n in names) + f" | {c['candidate_ratios']['Product']:.6f} | {c['duplicate_ratio']:.6f} |\n"
    text += '''
Aggregate duplicate ratios all meet [1/1.01, 1.01], and each worker meets
[1/1.02, 1.02]. Candidate aggregate means must improve by at least 2% against
**each** control at 30, padded-128 and 128; no case may regress over 1% in
aggregate or over 2% in a worker. The candidate fails those unchanged criteria.
These engineering screens are not confidence intervals. Passing duplicate-copy
checks does not establish that all layouts or time-order effects are controlled.
The retained records do not identify the cause of the differing product results.

| Tokens | Visit | CopyB / CopyA | Conditional / Product | Conditional / CopyA | Conditional / CopyB |
|---|---:|---:|---:|---:|---:|
'''
    for c in audit['cases']:
        for v in c['visits']:
            text += f"| {c['name']} | {v['visit']} | {v['copy_ratio']:.6f} | " + ' | '.join(f"{v['candidate_ratios'][n]:.6f}" for n in names[:3]) + ' |\n'
    text += '''
## Execution, arithmetic and code

Four fresh timing processes each run five cases. Each case has sixteen warmup
and forty-eight measured cycles with all four variants in balanced rotating
order; case order rotates and reverses across workers. Batches repeat the whole
bank 64/32/8/8/2 times at 8/30/padded-128/128/512 tokens. All 3,840 measured
batches and 1,280 warmup batches remain: 87,552 measured and 29,184 warmup bank
calls. Measured regions allocate zero bytes and record zero collections.
[All eighty per-worker/case/variant summaries](observations-20260920.json)
include means, medians, minima, maxima, allocation and GC counts.

An initial AMD code worker and every timing worker pass the same 1,575-case,
102,364,884-value bit comparison against the actual product, covering all sixty
captured layers plus geometry, offset, in-place, exceptional and random values.
Every timing batch matches independently pinned original census output bytes.
Inputs, biases and held expected outputs remain unchanged. These repeated finite
tests support exactness on the retained cases; they do not add fresh native
complete-model inference.

Actual optimized AMD assembly has 765 bytes for CopyA and 791 for Conditional.
The candidate's `vptest ymm8,ymm8` followed by `je G_M000_IG10` bypasses sixteen
vector FMAs and exponential reconstruction after six small-polynomial FMAs.
Both paths join before the same GELU output arithmetic/store. The full assembly
is retained, and three damaged opcode/target variants are rejected by the
machine-code checker.

All five workers exit 0. The 618 resource samples satisfy the fixed 3-GiB RSS,
600-second worker and 1-GiB available-memory limits. Maximum group RSS is
299,200,512 bytes; minimum available memory is 15,702,728,704 bytes. Timing
workers take 38.99–40.57 seconds. Recomputed foreign CPU snapshot fractions
range from 0.000509 to 0.000759 of four-CPU capacity during timing. Snapshots
miss some exited/short-lived activity and do not measure hypervisor contention.
Both supervisors and all five worker births are verified absent before closure.

## Evidence and reproduction

The [protocol and commands](README.md) were fixed before timing. The initial
local closure checker assumed a `SHORT` modifier on a jump that lacked one;
that diagnostic failure is preserved. Its corruption generator now matches
the actual optional modifier. No inference or successful writer was rerun.
Independent verification reproduces schedule counts, integer-total timing
means, every decision, original input/bias/output bank hashes, source/binary
identities and process accounting. Eight damaged real timing records and one
changed expected binary digest are refused, in addition to the assembly checks.

The closed local artifact includes the verified 120 input/bias files and their
original census receipt. Remote collection contains all 65 new source/result
files; already verified data/binaries are retained in the local payload. This
is an explicit verified subset of the original capture, not a claim that all
original captured outputs were transferred to AMD. All local files are bound
by the closure receipt. Successful writers are single-use.

'''
    identities = {'Measurement source': bundle['source_commit'], 'Unchanged qualified core SHA256': bundle['files']['bin/Lokad.Onnx.dll']['sha256'],
                  'Measurement host SHA256': bundle['files']['bin/Probe.dll']['sha256'], 'Frozen bundle SHA256': pin(base / 'payload/bundle.json')['sha256'],
                  'Collection SHA256': pin(base / 'collected/collection.json')['sha256'], 'Closed receipt SHA256': pin(base / 'closed.json')['sha256']}
    for name, value in identities.items():
        text += f'{name}: `{value}`.\n\n'
    text += 'Artifact: `artifacts/gelu-uniform-amd-20260920`. The failed screen supplies no basis to integrate or enable this shortcut.\n'
    observations = dict(audit, receipt=pin(base / 'closed.json'), verification=read(base / 'verification.json'), identities=identities)
    with targets[1].open('x', encoding='utf-8') as stream:
        json.dump(observations, stream, indent=2)
        stream.write('\n')
    with targets[0].open('x', encoding='utf-8') as stream:
        stream.write(text)
    print('Rendered all five cases, twenty visits and eighty timing summaries.')


if __name__ == '__main__':
    main()
