"""Publish the complete corrected case and explicit combined graph qualification."""
import csv
import json
from pathlib import Path
from qualified_graphs import BASE, E5, GRAPH, derive, pin, verified

OUT = Path(__file__).resolve().parent


def text(name, value):
    with (OUT/name).open('x', encoding='utf8') as stream:
        stream.write(value)


def main():
    assert not (OUT/'short-e5-20260925.md').exists()
    proof, combined = verified(BASE)
    actual, _ = derive()
    assert actual == combined
    e5_proof, e5 = verified(E5)
    row = e5['performance']
    lines = ['# Short-e5: one fixed warmup correction', '',
        '**Admission passes.**' if e5_proof['admitted'] else '**Admission fails.**', '',
        '| Release (seconds) | Relocation (seconds) | Microsoft ORT (seconds) | Relocation / ORT | Relocation / release |',
        '| ---: | ---: | ---: | ---: | ---: |',
        f"| {row['current']:.9f} | {row['candidate']:.9f} | {row['ort']:.9f} | {row['ratio']:.6f} | {row['candidate_over_current']:.6f} |", '',
        'The [exact-product runtime observation](../e5-relocation-tier-results/report-20260925.md)',
        'found optimized method loads after the old measured prefix. The prospective',
        'correction uses 6,000 fixed warmups and 180 measurements for every engine.',
        'Three numerical workers precede six fresh timing workers in release,',
        'candidate, ORT, ORT, candidate, release order. Every one of the 37,089 calls,',
        '1,080 measurements and nine setups is retained. No clock is trimmed.', '',
        'Core remains release f95a13c5 and relocation e07a4518. The compiled inspector',
        'confirms 65 of 66 consumer methods are unchanged; Main differs only in two',
        'count constants. All flags, public interfaces, branches, locals and exception',
        'regions remain equal. No profiler, product change or runtime override ran.', '',
        f"Repeatability: {sum(c['passed'] for c in row['controls'])}/3 controls pass at max/min <=1.10.",
        f"Candidate/release <=1.05: {row['regression_passed']}.",
        'All candidate outputs exactly match release bytes; fresh ORT scaled error',
        'is <=1e-4. Shape, finite-value, input and held-output ownership checks pass.', '',
        f"All {sum(r['samples'] for r in e5['resources']):,} resource samples pass; peak owned RSS is {max(r['peak_rss'] for r in e5['resources']):,} bytes.",
        'All 13 jobs and recorded process identities are terminal/code0.', '',
        '[All clocks](short-e5-clocks-20260925.csv), [setups](short-e5-setups-20260925.csv),',
        '[complete observations](short-e5-observations-20260925.json).',
        'The original failed graph campaign remains unchanged at def19d3f.',
        'Closure: `'+pin(E5/'closed.json')['sha256']+'`.']
    text('short-e5-20260925.md', '\n'.join(lines)+'\n')
    text('short-e5-clocks-20260925.csv', (E5/'clocks.csv').read_text())
    text('short-e5-observations-20260925.json', json.dumps(dict(closure=pin(E5/'closed.json'), **e5), indent=2)+'\n')
    with (OUT/'short-e5-setups-20260925.csv').open('x', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['process', 'seconds'])
        writer.writeheader()
        writer.writerows(e5['setups'])
    lines = ['# Dispatch relocation: explicitly combined graph qualification', '',
        '**All eight cases pass.**' if proof['admitted'] else '**Graph qualification remains incomplete.**', '',
        '| Case | Release (s) | Relocation (s) | Microsoft ORT (s) | Relocation / ORT | Warmups |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for result in combined['performance']:
        lines.append(f"| {result['key']} | {result['current']:.6f} | {result['candidate']:.6f} | {result['ort']:.6f} | {result['ratio']:.6f} | {result['warmups']} |")
    lines += ['', 'Seven cases retain their complete passing observations from def19d3f.',
        'Eight-token e5 uses the separately specified correction above. The original',
        'failed short-e5 row and its campaign remain immutable. This is an explicitly',
        'sourced qualification, not a reinterpretation of the old measurements.', '',
        'All cases use the same exact product binaries, models, inputs and numerical',
        'bounds. All retain 180 measured calls per timing process. These complete',
        'case observations contain 73,512 calls, 8,640 measurements and 72 setups;',
        'all 78,201 calls across both source campaigns remain retained.', '',
        f"Repeatability: {sum(c['passed'] for r in combined['performance'] for c in r['controls'])}/24 controls pass.",
        f"Regression: {sum(r['regression_passed'] for r in combined['performance'])}/8 gates pass.", '',
        '[Source protocols and full observations](qualified-graph-observations-20260925.json).',
        'The combined clock index is retained at artifacts/parakeet-owned-batch-graph-qualification-20260925/clocks.csv;',
        'its rows identify their source campaign. No source or BENCHMARK.md promotion',
        'follows: complete Parakeet, shared/Pyannote and package qualification remain.', '',
        'Original closure: `'+pin(GRAPH/'closed.json')['sha256']+'`.',
        'Correction closure: `'+pin(E5/'closed.json')['sha256']+'`.',
        'Combined closure: `'+pin(BASE/'closed.json')['sha256']+'`.']
    text('qualified-graphs-20260925.md', '\n'.join(lines)+'\n')
    text('qualified-graph-observations-20260925.json', json.dumps(dict(closure=pin(BASE/'closed.json'), **combined), indent=2)+'\n')
    print(json.dumps(dict(passed=True, short_e5_admitted=e5_proof['admitted'], graph_admitted=proof['admitted'])))


if __name__ == '__main__':
    main()
