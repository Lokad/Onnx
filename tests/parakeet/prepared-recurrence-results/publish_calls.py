"""Publish all actual decoder and complete-call qualification results from closure."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/parakeet-prepared-recurrence-calls-amd-v2-20260924'
OUT=Path(__file__).resolve().parent


def pin(path):
    with path.open('rb') as stream:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def read(path):return json.loads(path.read_text(encoding='utf8'))


def main():
    proof=read(BASE/'closed.json');assert proof['passed']
    for name,wanted in proof['files'].items():assert pin(BASE/name)==wanted,name
    analysis=read(BASE/'analysis.json');assert proof['analysis']==pin(BASE/'analysis.json')
    assert analysis['passed'] and analysis['no_performance_measurement'] and not analysis['root_product_changed']
    assert len(analysis['reviews'])==4 and len({r['output_digest'] for r in analysis['reviews']})==1
    samples=sum(r['samples'] for r in analysis['resources']);peak=max(r['peak_rss'] for r in analysis['resources'])
    value=dict(passed=True,closure=pin(BASE/'closed.json'),analysis=analysis,terminal=proof['remote_terminal'],
        first_refusal=pin(ROOT/'artifacts/parakeet-prepared-recurrence-calls-amd-20260924/closed.json'),publisher=pin(Path(__file__)))
    with (OUT/'calls-20260924.json').open('x',encoding='utf8') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    table='\n'.join(f"| {r['role']} | {r['mode']} | {r['exact_decoder_arrays']:,} | {r['exact_component_arrays']:,} | {r['max_native_error']:.10g} |" for r in analysis['reviews'])
    text=f'''# Prepared recurrence: actual decoder and complete calls

Candidate Core `3c23b44a` / Data `cc37b19e` passes actual decoder cache and execution
qualification in both instruction modes. All **6,080 original decoder arrays**
and **9,120 complete LSTM arrays** match the selected release exactly. Across
5,836,800 component values, maximum scaled error against Microsoft ORT1.29.0 is
**{analysis['max_native_error']:.10g}**, below the unchanged 1e-4 bound.

| Product | Instruction mode | Exact decoder arrays | Exact complete-call arrays | Maximum native error |
|---|---|---:|---:|---:|
{table}

Each worker repeats all 190 recorded decoder steps twice with independently
carried states, then repeats all 380 captured complete LSTM calls twice. Every
decoder token/duration decision stays exact. The original graph nodes, outputs,
13 initializer references and bytes remain unchanged. Input immutability and
held outputs survive later execution, reset and preparation invalidation.
One consumer `{analysis['consumer']['sha256']}` serves both products and modes.

The original decoder retains its three matrix preparations (25,246,720 bytes).
Candidate adds four immutable recurrent transposes (26,214,400 bytes), for
**51,461,120 bytes under the existing 64 MiB cap**. Each actual standalone LSTM
retains 13,107,200 candidate bytes versus zero selected. Every transpose element,
source/array ownership, original matrix digest, fresh-context sharing, repeated
preparation and invalidation/rebuild passes. Test-only poisoning of the internal
prepared arrays changes the actual decoder states and each standalone output to
NaN; rebuilding restores original bits. This proves the new arrays drive actual
execution while keeping the original weight bindings.

Independent NumPy review recomputes all raw arrays, native errors, cache receipts
and a common output digest. All **{samples} resource observations** pass; peak
owned RSS is **{peak:,} bytes**. Every PID/birth owner is terminal. No product or
native reference was rebuilt; canonical models and saved encoder arrays were
reused. [Machine-readable observations](calls-20260924.json) include every worker
receipt and resource summary. Raw evidence remains under
`artifacts/parakeet-prepared-recurrence-calls-amd-v2-20260924`.
Closure SHA-256: `{pin(BASE/'closed.json')['sha256']}`.

The first helper stopped on selected standalone input binding: it had declared
null inputs without shape descriptors. Refusal `bc2ce801` preserves that run;
no candidate ran. The revised helper supplies the exact captured float shapes.
Products, fixtures, gates and bounds are unchanged. See the
[prospective protocol](../prepared-recurrence-calls-amd-v2/README.md).

These are correctness results. Full native/public trajectories, prospectively
fixed component and application comparisons, and release regression gates remain
required. The selected release and BENCHMARK.md performance figures are unchanged.
'''
    with (OUT/'calls-20260924.md').open('x',encoding='utf8') as stream:stream.write(text)
    print(json.dumps(dict(report=pin(OUT/'calls-20260924.md'),observation=pin(OUT/'calls-20260924.json'))))


if __name__=='__main__':main()
