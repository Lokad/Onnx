"""Render the closed trial with this experiment title and recorded owner.

The inherited report.py was frozen but never executed; it retains two predecessor
display strings. This successor corrects those strings and labels the latency direction before reporting.
All scoring, numerical evidence and output fields remain unchanged.
"""
from fractions import Fraction
from pathlib import Path
import json
from protocol import pin, read, save
from score import ORDER

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'artifacts/pyannote-spatial-weight-screen-20260922'
OUT = Path(__file__).resolve().parent


def f(value): return Fraction(value['numerator'], value['denominator'])


def main():
    assert not (OUT/'results-20260922.md').exists() and not (OUT/'observations-20260922.json').exists()
    closed = read(BASE/'closed.json'); assert closed['passed']
    for name, wanted in closed['files'].items(): assert pin(BASE/name) == wanted, name
    a = read(BASE/'analysis.json')
    assert a['complete'] and a['numerical_and_resource_checks_pass'] and a['admitted'] == closed['admitted']
    census = read(ROOT/'artifacts/pyannote-blocked-spatial-census-20260922/census.json')
    reports = {name: read(BASE/'collected'/name/'result.json') for name in ORDER}
    save(OUT/'observations-20260922.json', dict(closure=pin(BASE/'closed.json'), analysis=a,
        processes=reports, forms=census['forms'], no_application_or_ort_speed_measurement=True))
    owner = read(BASE/'deployment.json')
    row = a['rows'][0]; p, c = f(row['production']), f(row['candidate'])
    failed_controls = [r for r in a['controls'] if not r['passed']]
    failed_gates = [r for r in a['gates'] if not r['passed']]
    lines = ['# AMD twelve-position weight reuse: complete-call screen', '',
        ('The component is **admitted for full product/application qualification**.' if a['admitted'] else 'The component is **not admitted**.'),
        f'All 108 call means sum to {float(p):.9f} s for selected production and',
        f'{float(c):.9f} s for the candidate: ratio {float(c/p):.9f}, {abs(1-float(c/p))*100:.4f}% {"lower" if c < p else "higher"}.',
        f'{len(failed_controls)} of 32 repeatability controls and {len(failed_gates)} of 12 speed gates fail.',
        'All numerical/resource checks pass. Selected root source is unchanged.', '',
        '| Form | Input C×H×W → output C×H×W | Stride | Calls per crop | Eligible | Production ms | Candidate ms | Ratio |',
        '|---:|---|---:|---:|---|---:|---:|---:|']
    for row in a['rows'][1:]:
        form = census['forms'][row['form']]
        shape = '×'.join(map(str,form['input_shape'][1:]))+' → '+'×'.join(map(str,form['output_shape'][1:]))
        lines.append(f'| {row["form"]} | {shape} | {form["attributes"]["strides"][0]} | {form["multiplicity"]} | {form["eligible"]} | {float(f(row["production"]))*1000:.6f} | {float(f(row["candidate"]))*1000:.6f} | {float(f(row["ratio"])):.6f} |')
    lines += ['', 'Each form includes its original multiplicity over three captured crops.',
        'All 108 calls and every fallback contribute. Four fresh ordinary processes',
        'run production, candidate, candidate, production on AMD EPYC 9V74 CPU2,',
        '.NET 10.0.8, without profiling. Geometry alone fixes 1,074 iterations per',
        'pass: one warmup and three measured passes, 17,184 clocks total, 4,296 warmup',
        'and 12,888 measured. Divide each call total by its iterations and three',
        'passes before summing means. No exclusions or sample trimming.', '',
        'Both exact products bind the same prepared ordinary graph caller. Timing',
        'includes inherited caller assertions/dispatch recording, finite scans,',
        'scratch, conversions, kernel and graph epilogues. Hash/journal IO is outside.',
        'This is not an embedding or diarization application timer and provides no',
        'new Microsoft ORT speed result.', '',
        '| Process | Sum of call means s | Graph creation/preparation, 32 nodes ms |',
        '|---|---:|---:|']
    for name in ORDER:
        lines.append(f'| {name} | {float(f(a["process_totals"][name]["total"])):.9f} | {float(f(a["preparation"][name]))*1000:.6f} |')
    lines += ['', 'All 512 separate graph creation/preparation clocks remain. These have a',
        'different boundary from older pure-helper preparation clocks. Both roles',
        'retain 21,086,208 fixture prepared bytes and 63,258,624 bytes in 108 independent',
        'call graphs after warmup. These fixtures do not measure whole-model residency.', '',
        'The unchanged scorer uses exact integer-clock fractions: process max/min',
        '≤1.10 aggregate and ≤1.20 per form; candidate/production ≤0.90 aggregate',
        'and ≤1.05 every eligible form. All gates are mandatory.']
    if failed_controls or failed_gates:
        lines += ['', 'Failed gates:', '']
        for category, rows in [('Repeatability',failed_controls),('Speed',failed_gates)]:
            for row in rows:
                lines.append(f'- {category}, {row.get("role","candidate")} form {row["form"] if row["form"] is not None else "aggregate"}: {float(f(row["ratio"])):.9f}, limit {float(f(row["limit"])):.2f}.')
    lines += ['', 'Exact selected Core `3c2f16b0` and candidate `3ca0a2a5` reuse qualified',
        'product and layer consumers. All output hashes match the native-checked',
        'reference; readonly operands, preparation hashes and dispatch checks pass.',
        'Twelve unchanged scorer tests and both untimed local driver checks pass.',
        f'All {a["samples"]} target resource observations pass; peak owned RSS {a["peak_rss"]:,} bytes.',
        f'Supervisor {owner["pid"]}/birth{owner["birth"]} and all four workers are terminal, exit 0.', '',
        f'Independent closure: `{pin(BASE/"closed.json")["sha256"]}`.',
        f'Analysis SHA-256: `{pin(BASE/"analysis.json")["sha256"]}`.',
        '[All clocks, preparation observations and gates](observations-20260922.json);',
        '[frozen prospective protocol](README.md);',
        '[AMD numerical proof](../spatial-weight-reuse-amd/results-20260922.md);',
        '[actual generated code](../spatial-weight-codegen/results-20260922.md).', '',
        ('Full product suites/package, Pyannote/Parakeet/shared/e5 regressions, long meetings and a fresh matched application/ORT campaign remain mandatory before integration.' if a['admitted'] else 'No root integration or unchanged timing retry follows this rejected screen.'),
        'Pyannote remains first, Parakeet second, Whisper deferred.']
    (OUT/'results-20260922.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(admitted=a['admitted'],report=pin(OUT/'results-20260922.md'),observations=pin(OUT/'observations-20260922.json'))))


if __name__ == '__main__': main()
