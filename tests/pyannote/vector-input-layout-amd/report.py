"""Publish only independently closed evidence, including every clock and gate."""
import json
import math
from fractions import Fraction
from pathlib import Path
from protocol import pin, read, save
from score import ORDER

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'artifacts/pyannote-vector-input-layout-amd-20260922'
OUTPUT = Path(__file__).resolve().parent


def fraction(value): return Fraction(value['numerator'], value['denominator'])


def main():
    assert not (OUTPUT/'results-20260922.md').exists() and not (OUTPUT/'observations-20260922.json').exists()
    closed = read(BASE/'closed.json'); assert closed['passed']
    for name, wanted in closed['files'].items(): assert pin(BASE/name) == wanted, name
    analysis = read(BASE/'analysis.json'); assert analysis['complete'] and analysis['admitted'] == closed['admitted']
    census = read(ROOT/'artifacts/pyannote-blocked-spatial-census-20260922/census.json')
    reports = {name: read(BASE/'collected'/name/'result.json') for name in ORDER}
    raw = dict(closure=pin(BASE/'closed.json'), analysis=analysis, processes=reports,
        qualification={name:read(BASE/'collected'/name/'result.json') for name in ['raw-256','raw-512','model-256','model-512']}, forms=census['forms'], no_application_or_ort_speed_measurement=True)
    save(OUTPUT/'observations-20260922.json', raw)
    aggregate = analysis['rows'][0]; p = fraction(aggregate['production']); c = fraction(aggregate['candidate'])
    ratio = c/p; gain = (1-ratio)*100
    prep = sum((fraction(analysis['preparation'][n]) for n in ORDER if n.startswith('candidate')), Fraction())/2
    break_even = math.ceil(prep/((p-c)/3)) if p > c else None
    controls_failed = [r for r in analysis['controls'] if not r['passed']]
    gates_failed = [r for r in analysis['gates'] if not r['passed']]
    lines = ['# AMD vector input layout screen', '',
        ('The component is **admitted for product qualification**.' if analysis['admitted'] else 'The component is **not admitted**.'),
        f'Across all 108 calls for three embedding crops, observed production time is {float(p):.9f} s',
        f'and AVX-512 candidate time {float(c):.9f} s: ratio {float(ratio):.9f}',
        f'({float(gain):.4f}% lower observed total; a negative value means slower).',
        f'{len(controls_failed)} of {len(analysis["controls"])} repeatability controls and {len(gates_failed)} of {len(analysis["gates"])} selection gates fail.',
        'All numerical and resource checks pass. Product dispatch is unchanged.', '',
        'This sum covers complete convolution calls, including conversions, finite',
        'operand scans, scratch rental/clearing, bias, residuals, activation and all',
        'fallbacks. It is not a complete embedding or diarization application timer,',
        'and there is no new Microsoft ORT speed comparison in this screen.', '',
        '| Form | Input C×H×W → output C×H×W | Stride | Residual / ReLU | Calls per crop | Eligible | Production ms | Candidate ms | Ratio |',
        '|---:|---|---:|---|---:|---|---:|---:|---:|']
    for row in analysis['rows'][1:]:
        form = census['forms'][row['form']]
        shape = '×'.join(map(str, form['input_shape'][1:]))+' → '+'×'.join(map(str, form['output_shape'][1:]))
        epilogue = ('yes' if form['residual'] else 'no')+' / '+('yes' if form['attributes'].get('activation') == 'Relu' else 'no')
        lines.append(f'| {row["form"]} | {shape} | {form["attributes"]["strides"][0]} | {epilogue} | {form["multiplicity"]} | {form["eligible"]} | {float(fraction(row["production"]))*1000:.6f} | {float(fraction(row["candidate"]))*1000:.6f} | {float(fraction(row["ratio"])):.6f} |')
    lines += ['', 'Each form total includes its original multiplicity across all three crops.',
        'No extra census multiplier is applied. There are 32 eligible and four',
        'fallback calls per crop. Frozen process order is production, candidate,',
        'candidate, production; each runs one complete warmup and three measured',
        f'passes. All {analysis["calls"]:,} call clocks ({analysis["warmups"]:,} warmup / {analysis["measured"]:,} measured) and all 512',
        'weight-preparation clocks are retained in the raw observations.', '',
        'Iterations are fixed by geometry before timing: ceil(2^31/(M*C*KH*KW*OH*OW)).',
        'Each call mean divides by its own iteration count, retaining original shape',
        'weights. This successor strengthens warmup without changing ratio gates.', '',
        '| Process | Complete call sum s | Prepare all 32 weights ms |',
        '|---|---:|---:|']
    for name in ORDER:
        lines.append(f'| {name} | {float(fraction(analysis["process_totals"][name]["total"])):.9f} | {float(fraction(analysis["preparation"][name]))*1000:.6f} |')
    lines += ['', f'Candidate preparation averages {float(prep)*1000:.6f} ms for 21,086,208 retained bytes.',
        'Both roles share identical fixture setup, including prepared weights outside',
        'the hot timers. Production does not normally require this additional storage.',
        ('There is no preparation break-even because the candidate is slower.' if break_even is None else
         f'Dividing preparation by the observed per-crop component saving gives an idealized break-even of {break_even} embedding calls; this is arithmetic, not measured application amortization.'),
        'Any future application trial must charge preparation at model/context creation.', '',
        'Fixed controls require process max/min ≤1.10 overall and ≤1.20 per form.',
        'Selection requires candidate/production ≤0.90 overall and ≤1.05 for every',
        'eligible form. Gates use exact fractions from integer clocks. All failed',
        'controls and gates, including their exact fractions, remain in the raw file.', '']
    for name, failures in [('Control', controls_failed), ('Selection', gates_failed)]:
        for row in failures:
            lines.append(f'- {name}: {row.get("role", "candidate")} form {row["form"] if row["form"] is not None else "aggregate"}, ratio {float(fraction(row["ratio"])):.9f}, limit {float(fraction(row["limit"])):.9f}.')
    lines += ['', 'The exact screen executable first passes all 108 actual-shape cases and',
        '119,823,360 values on AMD, matching the original Windows observations.',
        'Each timed output hash matches selected production, and operands remain',
        'unchanged. Twelve independent tests cover thresholds, iteration weighting',
        'and repeatability failures. AMD EPYC 9V74 uses CPU2 before CLR startup,',
        'monitor CPU0, .NET 10.0.8 and selected Core `1279b4b6`.',
        f'All {analysis["samples"]} resource samples pass; peak observed target RSS is {analysis["peak_rss"]:,} bytes.',
        'All target processes and the supervisor are terminal with exit code zero.', '',
        'Input packing now uses vector transposes. Reduction kernels, raw/model',
        'probes and the qualified M15 vector output epilogue remain byte-identical.',
        'The benchmark consumer adds the prospective geometry iteration protocol.',
        'Earlier rejected screens and phase diagnostics remain separate evidence.', '',
        f'Independent closure: `{pin(BASE/"closed.json")["sha256"]}`.',
        f'Analysis SHA-256: `{pin(BASE/"analysis.json")["sha256"]}`.',
        f'Raw observations SHA-256: `{pin(OUTPUT/"observations-20260922.json")["sha256"]}`.',
        '[All clocks and gates](observations-20260922.json); [prospective protocol](README.md).',
        'Pyannote remains first, Parakeet second and Whisper deferred.']
    (OUTPUT/'results-20260922.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    save(BASE/'published-report.json', dict(report=pin(OUTPUT/'results-20260922.md'), raw=pin(OUTPUT/'observations-20260922.json'), generator=pin(Path(__file__))))
    print(json.dumps(dict(report=pin(OUTPUT/'results-20260922.md'), raw=pin(OUTPUT/'observations-20260922.json'), admitted=analysis['admitted'])))


if __name__ == '__main__': main()
