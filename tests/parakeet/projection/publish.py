"""Render the closed projection evidence; never launches inference."""
from common import *


def main():
    closure = read(BASE/'closed.json')
    assert closure['qualified'] and all(absent(i) for i in closure['identities'])
    for name, expected in closure['files'].items(): assert pin(ROOT/name) == expected, name
    a = read(BASE/'analysis.json'); assert pin(BASE/'analysis.json') == closure['analysis']
    folder = Path(__file__).parent
    write(folder/'observations-20260921.json', a)
    lines = ['# Parakeet projection error on actual convolution outputs', '',
        'Completed on Windows i7-14700KF, logical CPU 2, on 2026-09-21. This',
        'numerical diagnostic uses the qualified managed core `d1f86a7346dc`',
        '(.NET 10.0.12, product `0f86c5d`) and Microsoft ONNX Runtime 1.29.0.',
        'It supplies no new latency measurement or production change.', '',
        'Both engines introduce projection error on identical captured inputs.',
        'The managed projection has about 4.8 times the local RMS error of ORT.',
        'Convolution error also exceeds the stem limit after the projection',
        'amplifies it. Correcting the projection alone cannot make this complete',
        'stem pass the existing `1e-4` limit against the independent references.', '',
        '## Measurement and controls', '',
        'The same retained English clip supplies two feature arrays, one from',
        'each frontend. Six full encoder calls cover both engines and both',
        'inputs, plus a repeat on native features for each engine. The graph',
        'adds only the stem and pre-projection tensor as outputs, with unknown',
        'dimensions; removing those outputs reproduces the original graph bytes.',
        'The unchanged frozen managed capture consumer is reused.', '',
        'All 69 checks pass: original full-encoder output, encoded length and',
        'stem are bit-identical; repeats are exact; inputs and held outputs are',
        'preserved; original optimized nodes are unchanged (2,856 managed,',
        '1,993 native, including native node attributes). No static-shape',
        'specialization or new fusion is used.', '',
        'Each captured `[1,74,4096]` input is independently multiplied by the',
        'original `[4096,1024]` FP32 weight and bias in float64 using NumPy 2.2.4',
        'with OpenBLAS and Torch 2.11.0+cpu with MKL. Both use one verified BLAS',
        'thread, exact promotion of the input/weights, and no intermediate FP32',
        'rounding. These are own-input references: they retain the original',
        'convolution rounding. Whole-stem references are reused from the',
        '[previous independent calculation](../stem-reference-v3/results-20260921.md).', '',
        f"All eight full-array reference comparisons pass `1e-9`; maximum scaled difference is `{max(r['max_scaled'] for r in a['agreements']):.12g}`.",
        f"All 2,048 independently recomputed scalar `math.fsum` dots pass; maximum scaled error is `{a['max_scalar_error']:.12g}`.", '',
        '## Local and inherited error', '',
        'Scaled error is `abs(actual-reference) / max(1,abs(reference))`.',
        'Every stem array has 75,776 values. Local error compares the original',
        'stem with a projection recomputed on its actual convolution input.',
        'Inherited error compares that recomputed projection with the entire',
        'double-precision stem. Total error compares the original with the',
        'entire double-precision stem. NumPy reference results follow; Torch',
        'gives the same failure counts with differences only near 1e-13.', '',
        '| Engine / feature source | Local max scaled | Local failures | Inherited max scaled | Inherited failures | Total max scaled | Total failures |',
        '|---|---:|---:|---:|---:|---:|---:|']
    rows = [r for r in a['rows'] if r['reference'] == 'numpy']
    labels = {'managed-native':'Lokad / native', 'native-native':'ORT / native',
              'managed-managed':'Lokad / managed', 'native-managed':'ORT / managed'}
    for row in rows:
        parts = [row['metrics'][k] for k in ('local-projection','inherited-convolution','total')]
        lines.append('| '+labels[row['route']]+' | '+' | '.join(f"{p['max_scaled']:.9g} | {p['failed']}" for p in parts)+' |')
    lines += ['', '| Engine / feature source | Local RMS | Inherited RMS | Total RMS |', '|---|---:|---:|---:|']
    for row in rows:
        lines.append('| '+labels[row['route']]+' | '+' | '.join(f"{row['metrics'][k]['rms']:.9g}" for k in ('local-projection','inherited-convolution','total'))+' |')
    lines += ['', 'The pre-projection arrays themselves all pass `1e-4` against their',
        'whole-stem reference inputs (maximum `2.958e-5`), yet their differences',
        'produce stem failures after projection. Local and inherited error',
        'vectors reconstruct total error exactly for all eight engine/reference',
        'pairs. RMS values and failure counts are not additive; the retained',
        '[observations](observations-20260921.json) include the interaction term.', '',
        '## Interpretation and follow-up', '',
        'The actual-input calculation establishes a projection accuracy issue',
        'independently of the frontend and convolutions. It also establishes',
        'that a perfect projection on these FP32 convolution outputs still',
        'fails the whole-stem reference limit. Both error sources need attention.',
        'The original three Windows duration-logit failures remain unresolved.', '',
        'Source inspection finds that reduction length 4096 is excluded from',
        'prepared weight packing by `GraphPacking.FitsPackBudget` (`n < 4096`).',
        'At 74 rows the runtime uses per-call packing and the two-row packed',
        'kernel, which carries each float accumulator through all 4096 terms.',
        'This source path suggests testing a more accurate accumulation method;',
        'it is not itself measured proof of a performance improvement. Any',
        'candidate needs actual-output, complete-model and performance tests.', '',
        '## Evidence and resources', '',
        'The eight fresh workers retain 24 capture arrays and 16 reference arrays.',
        f"All nine process identities are terminal. All {a['samples']} resource samples pass; peak RSS is {a['peak_rss']:,} bytes.",
        'Limits remain 900 seconds per worker, 8 GiB RSS, 10 GiB available RAM',
        'before a worker, 1 GiB available RAM during it, and 20 GiB free disk.',
        'The VM e5 campaign and queued Whisper timing lane are separate.', '',
        'Sources: `fe39ae6`. Artifact: `artifacts/parakeet-projection-20260921`.',
        f"Manifest: `{pin(BASE/'manifest.json')['sha256']}`.",
        f"Closure: `{pin(BASE/'closed.json')['sha256']}` ({len(closure['files'])} pinned files).", '',
        'The closure verifies arrays, actual libraries, controls, source identities,',
        'process/resource records and independently recomputed metrics. All',
        'previous evidence and numerical thresholds remain unchanged.', '']
    target = folder/'results-20260921.md'
    with target.open('x',encoding='utf8') as f: f.write('\n'.join(lines))
    print(json.dumps(dict(report=pin(target), observations=pin(folder/'observations-20260921.json'), files_verified=len(closure['files']))))


if __name__ == '__main__': main()
