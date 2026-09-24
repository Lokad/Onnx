"""Publish the completed native dispatch proof without repeating inference."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tests/parakeet/ort-diagnosis-amd'))
from run import pin, read


def main():
    base = ROOT/'artifacts/parakeet-ort-native-samples-20260924'
    match = ROOT/'artifacts/parakeet-ort-native-kernel-match-20260924'
    closed = read(base/'closed.json'); value = read(base/'analysis.json')
    assert closed['passed'] and closed['analysis'] == pin(base/'analysis.json')
    assert closed['auditor'] == pin(base/'audit-final.py')
    assert closed['kernel_match'] == pin(match/'closed.json')
    matched = read(match/'closed.json'); spec = read(match/'prepared.json')
    assert matched['passed'] and matched['analysis'] == pin(match/'analysis.json')
    assert matched['prepared'] == pin(match/'prepared.json')
    for name, identity in spec['files'].items(): assert pin(match/name) == identity
    assert (match/'candidate.bin').read_bytes() == (match/'original.bin').read_bytes()
    assert pin(match/'original.bin')['sha256'] == value['kernel']['sha256']
    graph_base = ROOT/'artifacts/parakeet-ort-graph-review-20260924'
    graph_closed = read(graph_base/'closed.json')
    assert graph_closed['passed']
    for name, identity in graph_closed['files'].items(): assert pin(graph_base/name) == identity
    out = Path(__file__).resolve().parent
    graphs = read(out/'ort-graphs-20260924.json')
    assert graphs['closure'] == pin(graph_base/'closed.json')
    groups = defaultdict(lambda: dict(nodes=0, seconds=0.0))
    for row in csv.DictReader((out/'ort-projections-20260924.csv').open(encoding='utf8')):
        if row['constant_b'] != 'True': continue
        group = groups[(int(row['k']), int(row['n']), row['op'], float(row['alpha']))]
        group['nodes'] += 1; group['seconds'] += float(row['corpus_seconds'])
    assert sum(row['nodes'] for row in groups.values()) == graphs['constant_projections']
    assert abs(sum(row['seconds'] for row in groups.values())-graphs['constant_projection_seconds']) < 1e-9
    kernel = value['kernel']
    compact = {k:v for k,v in value.items() if k not in ['instruction_addresses','per_request_period_ns']}
    compact.update(raw_artifact=str(base.relative_to(ROOT)).replace('\\','/'),
                   closure=pin(base/'closed.json'), raw_analysis=pin(base/'analysis.json'),
                   source_revision=spec['revision'], sources=spec['files'],
                   graph_review_closure=pin(graph_base/'closed.json'))
    lines = ['# Parakeet: observed ORT kernels and encoder projections', '',
        'The original ORT application spends **90.87% of measured sample weight**',
        'inside `MlasGemmFloatKernelAvx512F`. Its identity is proved against the',
        'installed binary: assembling the five matching Microsoft source files',
        'reproduces **all 8,904 bytes** of the sampled routine. No ORT replacement',
        'build or product change is used for inference.', '',
        '## What the actual graphs execute', '',
        'The serialized original-output graphs match every executed node name/type',
        'and original input/output descriptor: encoder 1,993 nodes, decoder 23,',
        'frontend 35. The encoder has 217 constant-weight matrix projections and',
        '72 matrix products with a runtime B operand. The constant projections',
        'cost **29.442083 profiled seconds per corpus**, versus **0.834040 seconds**',
        'for the dynamic products. Runtime shapes and all nodes remain retained.', '',
        '| Constant B shape (reduction × output columns) | ORT operator | Scale | Nodes | Profiled seconds/corpus |',
        '|---|---|---:|---:|---:|']
    for (k,n,op,alpha), group in sorted(groups.items(), key=lambda item:-item[1]['seconds']):
        lines.append(f"| {k} × {n} | {op} | {alpha:g} | {group['nodes']} | {group['seconds']:.6f} |")
    lines += ['',
        'All 217 constant projections expose only A in their runtime input profile;',
        'all 72 dynamic products expose both inputs. The matching ORT profiler lists',
        'allocated tensors; successful preparation releases constant tensors, and',
        '`MatMul<float>` consumes its prepared B buffer. `FusedMatMul` registers the',
        'same CPU implementation. Together these observations support prepared-B',
        'use; they are not a direct dump of native packing buffers.', '',
        'The 48 fused projections incorporate a scale of 0.5. Comparisons must',
        'include Lokad’s corresponding multiply and scale work. The serialized',
        'graphs preserve original outputs, avoiding the graph changes caused by',
        'earlier intermediate-output instrumentation. Temporary serialized weights',
        'were hashed and retired after session destruction; metadata remains.', '',
        '## Proof of native dispatch', '',
        f"Source revision: `{spec['revision']}`.",
        'Loaded library: `onnxruntime_pybind11_state.cpython-312-x86_64-linux-gnu.so`.',
        f"Library SHA256: `{spec['binary']['sha256']}`.",
        'ELF build ID: `3f20b61967f5eab0aff0b64364583da1845ca5e7`.',
        f"Matched interval: `{kernel['start']}` to `{kernel['end']}` (end exclusive).",
        f"Routine SHA256: `{kernel['sha256']}`.", '',
        'The installed library has no internal symbols. The assembly match gives',
        'the function name; the nearest exported Python entry-point label does not.',
        'Sampled hot instructions include its 12-row × 32-column block: two B vectors',
        'are shared across 24 AVX-512 accumulators, with four reduction steps unrolled',
        'and explicit prefetches. The routine also contains smaller-row paths.',
        'The 90.87% share covers the whole function, not just that block.', '',
        '| Capture check | Observed |', '|---|---:|',
        f"| Raw / measured-request samples | {value['total_samples']:,} / {value['measured_samples']:,} |",
        f"| Lost samples | {value['raw']['lost']} |",
        f"| Sample-period coverage of measured wall time | {100*value['coverage_ratio']:.3f}% |",
        f"| Complete sampled corpus | {value['corpus_seconds']:.6f}s |",
        f"| Wall change from boundary control | {100*(value['over_control']-1):+.3f}% |",
        f"| Function share of measured sample periods | {100*kernel['share']:.3f}% |",
        f"| Sampled function seconds/corpus, estimate | {kernel['estimated_seconds_per_corpus']:.6f}s |",
        f"| Peak owned RSS | {value['peak_rss']:,} bytes |", '',
        'One warmup and three measured passes preserve all 80 original request',
        'checks. Samples are joined to the 60 measured public-request intervals.',
        'Every exported event is reconciled to its raw PID, thread, timestamp and',
        'instruction address. Three measured samples belong to auxiliary threads;',
        'one main-thread event has no unwound stack and retains its raw instruction',
        'address. No event is dropped. The initial parser assumptions and correction',
        'are recorded separately; no inference was repeated to repair analysis.', '',
        'This function serves other GEMMs, including convolution and decoder work.',
        'Its sample weight must not all be assigned to encoder constant projections.',
        'Operator timings come from a separate capture with 1.50% additional wall',
        'time; sampling adds 0.61% relative to the boundary control. These differences',
        'also include process variation. No overhead is subtracted.', '',
        '## Narrowed next comparison', '',
        'The selected Lokad profile assigns about 37.2% to `ShortWideMultiply2Rows`',
        'and `ShortWideMultiply3Rows`, 9.5% to another AVX2 packed kernel, 6.1% to',
        'packing and 3.1% to `PackedTile12`. Lokad already has a 12-row AVX-512',
        'kernel, enabled for eligible prepared weights. Its per-call path uses',
        'the two-/three-row kernels by default. The encoder retains 37 constant',
        'weights at its 256 MiB cap. Row eligibility and residency both affect',
        'dispatch; the ORT constant count alone predicts no saving.', '',
        'Measure selected Lokad phase clocks and the complete matching projection',
        'groups, including packing, scale and destination handling. Join their',
        'actual dispatch to the exact shapes, then compare generated loops against',
        'the proved native routine. Rank excess seconds before selecting one change.',
        'Reject a projection-focused optimization hypothesis if these complete',
        'groups do not explain the largest excess, or if the proposed route already',
        'executes with comparable complete-call cost. A repeat of the rejected',
        'global AVX-512 toggle or packing-boundary trial is not justified.', '',
        'The maximum defensible saving is bounded by measured excess in the affected',
        'groups, which is still unknown. Tile size does not predict a speedup.',
        'The qualified release remains 73.82s versus ORT 39.61s, ratio 1.864.', '',
        '[Phase/operator clocks](ort-phases-20260924.md),',
        '[every encoder projection](ort-projections-20260924.csv),',
        '[graph/source review](ort-graphs-20260924.json),',
        '[native sample identities and accounting](ort-kernel-observations-20260924.json).', '',
        f"Native closure SHA256: `{pin(base/'closed.json')['sha256']}`.",
        f"Kernel-match closure SHA256: `{pin(match/'closed.json')['sha256']}`.", '']
    with (out/'ort-kernel-observations-20260924.json').open('x', encoding='utf8') as stream:
        json.dump(compact, stream, indent=2); stream.write('\n')
    with (out/'ort-kernels-20260924.md').open('x', encoding='utf8') as stream:
        stream.write('\n'.join(lines))
    print(json.dumps(dict(report=pin(out/'ort-kernels-20260924.md'),kernel=kernel)))


if __name__ == '__main__': main()
