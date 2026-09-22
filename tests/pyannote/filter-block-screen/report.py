"""Report the closed four-block trial without changing its observations or gates."""
from fractions import Fraction
from pathlib import Path
import json
from protocol import pin, read, save
from score import ORDER

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'artifacts/pyannote-filter-block-screen-20260922'
OUTPUT = Path(__file__).resolve().parent


def fraction(value): return Fraction(value['numerator'], value['denominator'])


def main():
    assert not (OUTPUT/'results-20260922.md').exists()
    assert not (OUTPUT/'observations-20260922.json').exists()
    closed = read(BASE/'closed.json')
    assert closed['passed'] and not closed['admitted']
    assert pin(BASE/'closed.json')['sha256'] == '553e00fe969f05be3025db0afb62b504380e770a6d851413b962701e9b96d88e'
    for name, wanted in closed['files'].items(): assert pin(BASE/name) == wanted, name
    analysis = read(BASE/'analysis.json')
    assert analysis['complete'] and analysis['numerical_and_resource_checks_pass'] and not analysis['admitted']
    census = read(ROOT/'artifacts/pyannote-blocked-spatial-census-20260922/census.json')
    reports = {name: read(BASE/'collected'/name/'result.json') for name in ORDER}
    save(OUTPUT/'observations-20260922.json', dict(closure=pin(BASE/'closed.json'),
        analysis=analysis, processes=reports, forms=census['forms'],
        no_application_or_ort_speed_measurement=True))
    aggregate = analysis['rows'][0]
    p, c = fraction(aggregate['production']), fraction(aggregate['candidate'])
    lines = ['# AMD four-block convolution: rejected complete-call screen', '',
        f'The candidate is **not admitted**. The sum of all 108 call means is {float(p):.9f} s',
        f'for selected production and {float(c):.9f} s for the candidate, ratio {float(c/p):.9f}.',
        f'The {(1-float(c/p))*100:.4f}% reduction misses the fixed 10% component gate.',
        'Forms 1 and 2 also exceed their 5% regression limit. All 32 repeatability',
        'controls pass; three of twelve selection gates fail. Numerical and',
        'resource checks pass. Selected product source remains unchanged.', '',
        '| Form | Input C×H×W → output C×H×W | Stride | Calls per crop | Eligible | Production ms | Candidate ms | Ratio |',
        '|---:|---|---:|---:|---|---:|---:|---:|']
    for row in analysis['rows'][1:]:
        form = census['forms'][row['form']]
        shape = '×'.join(map(str, form['input_shape'][1:]))+' → '+'×'.join(map(str, form['output_shape'][1:]))
        lines.append(f'| {row["form"]} | {shape} | {form["attributes"]["strides"][0]} | {form["multiplicity"]} | {form["eligible"]} | {float(fraction(row["production"]))*1000:.6f} | {float(fraction(row["candidate"]))*1000:.6f} | {float(fraction(row["ratio"])):.6f} |')
    lines += ['', 'Each form total includes its original multiplicity across three captured crops.',
        'Both roles execute the same prepared ordinary graph caller. Complete-call',
        'timers include caller assertions and dispatch recording, finite scans, scratch',
        'rentals, conversions, convolution, graph epilogues and every fallback.',
        'Output hashing and journal IO are outside. This is a component comparison;',
        'there is no new complete application or Microsoft ORT timing measurement.', '',
        'Four fresh AMD EPYC 9V74 CPU2 processes run production, candidate, candidate,',
        'production, .NET 10.0.8 with ordinary settings and no profiler. Fixed geometry',
        'determines iterations before execution: 1,074 per pass, one warmup and three',
        'measured passes. All 17,184 call clocks remain: 4,296 warmup and 12,888 measured.',
        'Per-call means divide measured totals by their iterations and three passes;',
        'summing those means preserves the original 108 call weights.', '',
        '| Process | Sum of call means s | Graph creation and preparation, 32 nodes ms |',
        '|---|---:|---:|']
    for name in ORDER:
        lines.append(f'| {name} | {float(fraction(analysis["process_totals"][name]["total"])):.9f} | {float(fraction(analysis["preparation"][name]))*1000:.6f} |')
    lines += ['', 'All 512 separate preparation clocks remain. These include actual graph',
        'creation and weight preparation; they have a different boundary from older',
        'pure-helper preparation measurements. Both roles have 21,086,208 fixture',
        'prepared-weight bytes and 63,258,624 bytes retained by 108 independent call',
        'graphs after warmup. These are component fixtures, not whole-model residency.', '',
        'Exact integer-clock fractions decide every gate, with no rounding, trimming',
        'or exclusions. Controls require max/min ≤1.10 aggregate and ≤1.20 per form.',
        'Selection requires candidate/production ≤0.90 aggregate and ≤1.05 for every',
        'eligible form. Failed selection gates:', '']
    for row in analysis['gates']:
        if not row['passed']:
            lines.append(f'- Form {row["form"] if row["form"] is not None else "aggregate"}: {float(fraction(row["ratio"])):.9f}, limit {float(fraction(row["limit"])):.2f}.')
    lines += ['', 'Selected Core `3c2f16b0` and candidate `64f42e97` reuse their qualified',
        'product and layer-consumer assemblies unchanged. Every timed output hash',
        'matches the native-checked layer reference; read-only operands remain exact.',
        'Twelve inherited scorer tests and both local driver validations pass.',
        f'All {analysis["samples"]} target resource observations pass, peak owned RSS {analysis["peak_rss"]:,} bytes.',
        'Supervisor 715202/birth1790090292.55 and all four workers are terminal, exit 0.', '',
        'The [numerical qualification](../filter-block-reuse-amd/results-20260922.md)',
        'and [generated-code inspection](../filter-block-codegen/results-20260922.md)',
        'remain valid but do not override this failed performance decision. The',
        'optimized four-block kernel keeps its vectors in registers, yet that alone',
        'does not establish a useful complete-call gain. Repeated input-address',
        'arithmetic is a separate source-guided hypothesis, not a proven cause.', '',
        f'Closure SHA-256: `{pin(BASE/"closed.json")["sha256"]}`.',
        f'Analysis SHA-256: `{pin(BASE/"analysis.json")["sha256"]}`.',
        '[All clocks, gates and preparation observations](observations-20260922.json);',
        '[frozen prospective protocol](README.md).', '',
        'Pyannote remains first, Parakeet second and Whisper deferred. The admitted',
        'application result stays 13.165243357 s versus ORT 8.952561215 s, ratio 1.470556.',
        'No unchanged retry or integration of this rejected candidate follows.']
    (OUTPUT/'results-20260922.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    print(json.dumps(dict(report=pin(OUTPUT/'results-20260922.md'), observations=pin(OUTPUT/'observations-20260922.json'))))


if __name__ == '__main__': main()
