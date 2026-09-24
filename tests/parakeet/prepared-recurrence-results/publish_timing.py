"""Publish the frozen complete-call verdict, all controls and exact timing means."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/parakeet-prepared-recurrence-timing-amd-20260924'
OUT=Path(__file__).resolve().parent


def pin(path):
    with path.open('rb') as stream:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def read(path):return json.loads(path.read_text(encoding='utf8'))


def main():
    proof=read(BASE/'closed.json');assert proof['passed']
    for name,wanted in proof['files'].items():assert pin(BASE/name)==wanted,name
    analysis=read(BASE/'analysis.json');assert proof['analysis']==pin(BASE/'analysis.json')
    performance=analysis['performance'];assert proof['admitted']==performance['admitted']
    raw={};setup=[]
    for worker in analysis['reviews']:
        name=next(r['name'] for r in analysis['resources'] if (BASE/'collected'/r['name']/'output/result.json').is_file() and pin(BASE/'collected'/r['name']/'output/result.json')==worker['result'])
        path=BASE/'collected'/name/'output/result.json';result=read(path);raw[path.relative_to(ROOT).as_posix()]=pin(path)
        setup.append(dict(worker=name,seconds=sum(r['ticks'] for r in result['setup'])/result['frequency'],
            allocated_bytes=sum(r['allocated_bytes'] for r in result['setup']),retained_bytes=sum(r['retained_bytes'] for r in result['setup'])))
    value=dict(passed=True,closure=pin(BASE/'closed.json'),analysis=analysis,raw_results=raw,setup=setup,
        terminal=proof['remote_terminal'],publisher=pin(Path(__file__)))
    with (OUT/'timing-20260924.json').open('x',encoding='utf8') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    def number(value):return float(Fraction(**value))
    rows='\n'.join(f"| {r['mode']} | {r['group']} | {number(r['selected'])*1000:.6f} | {number(r['candidate'])*1000:.6f} | {number(r['candidate_over_selected']):.6f} |" for r in performance['table'])
    cold='\n'.join(f"| {r['worker']} | {r['seconds']*1000:.6f} | {r['allocated_bytes']:,} | {r['retained_bytes']:,} |" for r in setup)
    bad_controls=sum(not r['passed'] for r in performance['controls']);bad_gates=sum(not r['passed'] for r in performance['gates'])
    verdict='Admitted to the complete application comparison.' if performance['admitted'] else ('Comparison invalid: repeatability controls failed.' if not performance['controls_passed'] else 'Rejected at the fixed component performance gates.')
    samples=sum(r['samples'] for r in analysis['resources']);peak=max(r['peak_rss'] for r in analysis['resources'])
    text=f'''# Prepared recurrence: complete captured-call timing

**{verdict}** Failed controls: **{bad_controls}/84**. Failed performance gates:
**{bad_gates}/14**. All numerical, identity, ownership and resource checks pass.
The selected release and BENCHMARK.md remain unchanged.

| Mode | Captured case | Selected total ms | Candidate total ms | Candidate / selected |
|---|---|---:|---:|---:|
{rows}

Each row is the exact mean of ten measured case passes across two fresh
processes. The corpus contains all 380 complete LSTM calls in their captured
case/step/node order. Both instruction modes use selected0, candidate0,
candidate1, selected1; every process has five fixed warmup and five measured
passes. All **30,400 raw call clocks**, including 15,200 warmup clocks, remain
retained. No trimming or adaptive warmup was used. All 91,200 output-array hashes
match the qualified capture; inputs/constants and held outputs remain unchanged.

Clocks include reset, ordinary graph validation, allocation and complete LSTM
execution. Caller fixture/feed creation, output hashing and serialization are
outside each clock. Separate cold setup includes both graph constructions,
preparations and execution-context creation:

| Worker | Cold setup ms | Thread allocated bytes | Retained prepared bytes |
|---|---:|---:|---:|
{cold}

These two setup observations per product/mode are not a precise cold-start
latency estimate. Candidate keeps four immutable arrays totaling 26,214,400
bytes across the two graphs; each graph stays under its unchanged 64 MiB cap.

All **{samples} resource observations** pass; peak owned RSS is **{peak:,} bytes**.
Every worker/thread uses AMD CPU2, with monitoring on CPU0 and .NET10.0.8.
Observed foreign CPU fractions are <=0.01; snapshot accounting can miss some
short-lived processes, and its limitation is retained. Every PID/birth owner
is terminal. [All exact means, controls and receipts](timing-20260924.json) link
the raw results under `artifacts/parakeet-prepared-recurrence-timing-amd-20260924`.
Closure SHA-256: `{pin(BASE/'closed.json')['sha256']}`.

The [prospective protocol](../prepared-recurrence-timing-amd/README.md) requires
all 84 repeatability controls, at least 10% corpus gain in both modes, and no case
over 5% slower. It forbids unchanged retries. Component results do not establish
end-to-end parity. The separate full application comparison still requires all
63 controls, at least 3% corpus gain and no clip over 5% slower; release regression
gates follow any application admission.
'''
    with (OUT/'timing-20260924.md').open('x',encoding='utf8') as stream:stream.write(text)
    print(json.dumps(dict(report=pin(OUT/'timing-20260924.md'),observations=pin(OUT/'timing-20260924.json'),admitted=performance['admitted'])))


if __name__=='__main__':main()
