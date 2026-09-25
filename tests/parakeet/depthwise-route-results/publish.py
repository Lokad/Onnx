"""Publish completed managed counters and exact native kernel evidence once."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def read(p): return json.loads(p.read_text(encoding='utf8'))


def pin(p):
    with p.open('rb') as f: return dict(bytes=p.stat().st_size, sha256=hashlib.file_digest(f, 'sha256').hexdigest())


def main():
    base = ROOT/'artifacts/parakeet-depthwise-route-amd-20260925'
    native = ROOT/'artifacts/parakeet-ort-depthwise-kernels-20260925'
    closed = read(base/'closed.json'); counts = read(base/'analysis.json')
    assert closed['passed'] and closed['analysis'] == pin(base/'analysis.json')
    raw = base/'capture-collected/logs/counts.json'
    assert counts['passed'] and counts['raw_counts'] == pin(raw)
    native_closed = read(native/'corrected-closed.json'); kernels = read(native/'corrected-analysis.json')
    assert native_closed['passed'] and native_closed['analysis'] == pin(native/'corrected-analysis.json')
    for name, wanted in native_closed['files'].items(): assert pin(native/name) == wanted
    assert kernels['passed'] and all(k['observed'] for k in kernels['matches'])
    rows = read(raw)['rows']; groups = defaultdict(lambda:defaultdict(int))
    for row in rows:
        label = 'stem' if row['geometry'][1] == 256 else 'modules'
        for field in ['calls', 'panels', 'products', 'views', 'patch_values']:
            assert row[field] % 4 == 0
            groups[label][field] += row[field]//4
        for layout in row['layouts']:
            for pair in layout.split(' | '):
                before, after = pair.split(' -> ')
                assert before == after and before.startswith('Lokad.Onnx.DenseTensor`1[System.Single]:False:')
    report = dict(passed=True, diagnostic_only=True, release_admitted=False,
                  managed_closure=pin(base/'closed.json'), native_closure=pin(native/'corrected-closed.json'),
                  managed=counts, per_corpus_groups=dict(groups), all_geometry_counters=rows,
                  native=kernels, original_native_sample_capture='2026-09-24',
                  native_samples_not_node_exclusive=True, no_new_native_inference=True)
    with (OUT/'diagnosis-20260925.json').open('x', encoding='utf8') as f:
        json.dump(report, f, indent=2, allow_nan=False); f.write('\n')
    lines = ['# Parakeet depthwise convolution: observed cause and one next experiment', '',
        'The complete count capture confirms the predicted managed mechanism for',
        'every one of the **59 actual geometries**. Across one 20-clip corpus, M78',
        'constructs **11,845,632 tensor views**, dispatches **3,948,544 tiny matrix',
        'products**, and writes **1,084,870,656 float values** into expanded patch',
        'buffers. All actual matrix leaves are the one-row FMA path. The inputs and',
        'weights are already ordinary dense tensors before materialization.', '',
        '| Observed work per corpus | Two stem nodes | 24 module nodes |',
        '| --- | ---: | ---: |']
    for name, field in [('Operator calls','calls'),('Panels','panels'),('Matrix products','products'),
                        ('Tensor views','views'),('Patch float values','patch_values')]:
        lines.append(f"| {name} | {groups['stem'][field]:,} | {groups['modules'][field]:,} |")
    lines += ['',
        'The rented scratch arrays are 524,288 bytes for the stem and 2,097,152',
        'bytes for the modules, versus requests of 327,680 and 1,310,720 bytes.',
        'These are physical array lengths, not accumulated allocation or DRAM traffic.',
        'Each partial panel, product dimension and layout is retained in the JSON.', '',
        '## Exact ORT kernels now observed', '',
        'The original native samples contain instruction addresses inside both',
        'predicted functions. Their identities are verified against the installed',
        'library, SHA256 `ff54b93f257508c8e32a43f3528382b7eb791f0946730187e4767d7292a290ee`.',
        'This used the exact installed source revision',
        '`2e2543fbe9fae542f921d47a72d21d5a4ef0b710`, with no new native inference.', '',
        '| Function | Installed file interval | Distinct sampled addresses | Sample-period estimate (s/corpus) |',
        '| --- | --- | ---: | ---: |']
    for k in kernels['matches']:
        lines.append(f"| `{k['target']}` | `{k['start']:#x}..{k['end']:#x}` | {len(k['sampled_instruction_addresses'])} | {k['estimated_seconds_per_corpus']:.6f} |")
    lines += ['',
        'All 53,162 bytes of the convolution object text match uniquely after resolving',
        '52 internal calls. Its depthwise entry and local helper occupy 608 bytes;',
        'shared postprocessing functions outside that interval are excluded from its',
        'sample estimate. All 652 M1 bytes match after resolving two PC-relative',
        'references to the exact eight mask constants. The initial matcher stopped',
        'on unsupported internal calls; that failure is retained and its assembled',
        'object was reused. Five identity/rejection tests pass.', '',
        'The stem uses blocked-channel NCHWc and a dedicated direct depthwise kernel.',
        'The later 1D nodes use ordinary convolution with segmented expansion and',
        'single-row matrix products. The optimized graph and exact dispatch source',
        'connect these algorithms to the observed shapes. Native samples establish',
        'that the kernels execute somewhere in the original public workload; they',
        'do not give per-node invocation counts or prove exclusive encoder attribution.',
        'In particular, the M1 sample estimate excludes expansion and dispatch costs.', '',
        '## Bounded optimization hypothesis', '',
        'The closed operator profile assigns these 26 nodes **2.962622 s in M78',
        'versus 0.194421 s in ORT**, a **2.768202-second excess**. The stem pair',
        'contributes 1.534429 versus 0.043986 seconds; modules contribute 1.428194',
        'versus 0.150435 seconds. These costs exclude separate Pad and activation',
        'operators. They are diagnostic profile clocks, not a new application score.', '',
        'Select one candidate mechanism: direct nine-tap depthwise accumulation in',
        'the existing dense layout, bypassing patch materialization, per-channel',
        'tensor views and generic matrix dispatch for the two observed geometries.',
        'Preserve reduction order, sum-then-bias, SIMD/scalar options and fallback',
        'contracts. ORT supplies evidence for avoiding generic overhead; copying',
        'its blocked layout would require a separate, broader graph change.', '',
        'The falsifiable mechanism prediction is zero patch values, temporary views',
        'and generic matrix calls for all 520 eligible operator calls per corpus.',
        'At the isolated application baseline of 55.835237 seconds, the existing',
        '3% whole-application requirement needs at least 1.675057 seconds saved.',
        'Under additive unchanged-other-work assumptions, these operators must cost',
        'at most 1.287565 seconds. A shape-weighted screen can reject insufficient',
        'benefit; only a fresh complete application comparison can establish the gain.',
        'Do not sweep tile sizes, instruction sets or nearby arithmetic variants.', '',
        '## Validation and release boundary', '',
        'The counter-only build changes three methods; 3,274 original Core methods,',
        'all 697 Data methods and the public consumer remain unchanged. There are',
        'no added build warnings. All 80 public results match exactly, including',
        'transcripts, tokens and decoder calls. All 457 resource observations pass;',
        'peak owned RSS is 9,219,780,608 bytes. Instrumented elapsed times are unscored.', '',
        'The candidate is planned, not implemented or admitted. The qualified root',
        'product and BENCHMARK.md stay unchanged; M78 remains isolated because of',
        'its independent e5 regression. This depthwise target cannot by itself close',
        'the complete Parakeet gap to ORT.', '',
        '[All counters and native evidence](diagnosis-20260925.json),',
        '[matched stem source/shapes](../stem-diagnosis/diagnosis-20260925.md),',
        '[native identity method](../depthwise-native-proof/README.md).', '',
        f"Managed closure: `{report['managed_closure']['sha256']}`.",
        f"Native closure: `{report['native_closure']['sha256']}`.", '']
    with (OUT/'diagnosis-20260925.md').open('x', encoding='utf8') as f: f.write('\n'.join(lines))
    print(json.dumps(dict(report=pin(OUT/'diagnosis-20260925.md'), evidence=pin(OUT/'diagnosis-20260925.json'))))


if __name__ == '__main__': main()
