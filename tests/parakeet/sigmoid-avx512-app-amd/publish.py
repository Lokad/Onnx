"""Publish the closed application verdict without executing or rescoring requests."""
import json
from pathlib import Path
from prepare import BASE, ROOT
from protocol import pin, read


def main():
    proof = read(BASE/'closed.json')
    assert proof['passed']
    for name, wanted in proof['files'].items():
        assert pin(BASE/name) == wanted, name
    analysis = read(BASE/'analysis.json')
    assert proof['analysis'] == pin(BASE/'analysis.json')
    assert analysis['passed'] and analysis['reference_provenance_verified']
    assert not analysis['root_product_changed']
    assert (analysis['timing_requests'], analysis['warmup'], analysis['measured']) == (480, 120, 360)
    table = analysis['table']; performance = analysis['performance']
    assert len(table) == 21 and len(performance['controls']) == 63 and len(performance['gates']) == 21
    assert proof['admitted'] == performance['admitted']
    corpus = next(row for row in table if row['is_corpus'])
    assert corpus['name'] == 'complete-corpus' and corpus['audio_seconds'] == 213.265
    assert all(len(corpus[role]['processes']) == 2 for role in ['current', 'candidate', 'ort'])
    report = dict(application_admitted=performance['admitted'], release_admitted=False,
        application_closure=pin(BASE/'closed.json'), full_analysis=pin(BASE/'analysis.json'),
        identities=analysis['identities'], consumers=analysis['consumers'],
        requests=analysis['timing_requests'], warmup=analysis['warmup'], measured=analysis['measured'],
        table=table, performance=performance, results=analysis['results'], resources=analysis['resources'],
        root_product_changed=False, prediction_corpus_seconds_saved=0.45,
        observed_corpus_seconds_saved=corpus['current']['seconds']-corpus['candidate']['seconds'],
        prediction_met=corpus['current']['seconds']-corpus['candidate']['seconds']>=0.45, source=pin(Path(__file__)))
    target = ROOT/'tests/parakeet/sigmoid-avx512-results/application-20260928.json'
    target.parent.mkdir(exist_ok=True)
    with target.open('x', encoding='utf8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    markdown=target.with_suffix('.md')
    controls=sum(r['passed'] for r in performance['controls'])
    gates=sum(r['passed'] for r in performance['gates'])
    current,candidate,native=[corpus[role]['seconds'] for role in ['current','candidate','ort']]
    gain=100*(1-candidate/current)
    ratio=corpus['ratios_to_ort']['candidate']
    verdict='admitted' if performance['admitted'] else 'not admitted'
    parity='met' if performance['parity_target_met'] else 'not met'
    lines=['# Complete Parakeet result for the fixed AVX-512 sigmoid','',
        f'The isolated candidate is **{verdict}** by the original application rules.',
        f'Current {current:.9f}s; candidate {candidate:.9f}s; Microsoft ORT {native:.9f}s.',
        f'Matched candidate gain: {gain:.6f}%. Candidate / ORT: {ratio:.9f}.',
        f'Repeatability controls: {controls}/63. Gain and clip-regression gates: {gates}/21.',
        f'The separate <=1.05 parity target is {parity}.','',
        'Twenty clips total 213.265 seconds of audio. Six fresh processes run',
        'current/candidate/ORT/ORT/candidate/current, one warmup and three measured',
        'passes per clip. All 480 requests / 360 measured clocks are retained.',
        'Every output, complete retained result, input identity and resource check passes.',
        'No clock is trimmed or corrected, and no failed gate is overridden.','',
        'One fixed AVX-512 rational sigmoid loop follows the verified native ORT',
        'routine and retained managed instruction samples. The portable helper,',
        'Data, graph structure, allocations, consumer and runtime flags are unchanged.',
        'All focused and complete-model correctness gates passed before scoring.',
        f'Prospective saving: 0.45 seconds; measured saving: {current-candidate:.9f} seconds.',
        f'Prediction met: {current-candidate >= 0.45}.','',
        '| Clip | Current, s | Candidate, s | ORT, s | Candidate / current | Candidate / ORT |',
        '|---|---:|---:|---:|---:|---:|']
    for row in table:
        a,b,c=[row[role]['seconds'] for role in ['current','candidate','ort']]
        lines.append(f"| {row['name']} | {a:.9f} | {b:.9f} | {c:.9f} | {b/a:.9f} | {row['ratios_to_ort']['candidate']:.9f} |")
    lines.extend(['','Failed repeatability controls:'])
    failed=[r for r in performance['controls'] if not r['passed']]
    lines+=['']+([f"- {r['name']} / {r['role']}: {r['process_ratio']:.9f} > {r['limit']:.2f}." for r in failed] or ['None.'])
    lines.extend(['','Failed gain or regression gates:',''])
    failed=[r for r in performance['gates'] if not r['passed']]
    lines+=([f"- {r['name']}: candidate/current {r['candidate_over_current']:.9f} > {r['limit']:.2f}." for r in failed] or ['None.'])
    lines.extend(['',f"Closure: {pin(BASE/'closed.json')['sha256']}.",
        '[Complete observations and every clock](application-20260928.json).',
        'Artifacts: artifacts/parakeet-sigmoid-avx512-app-amd-20260928.','',
        'This is an isolated application verdict. Root source and BENCHMARK.md remain',
        'unchanged until broader release qualification. A rejected result remains',
        'rejected; any further experiment requires a specific explained cause.',''])
    with markdown.open('x',encoding='utf8') as stream:stream.write('\n'.join(lines))
    print(json.dumps(dict(published=pin(target), application_admitted=performance['admitted'],
        controls_passed=sum(r['passed'] for r in performance['controls']),
        gates_passed=sum(r['passed'] for r in performance['gates']),
        corpus={role:corpus[role]['seconds'] for role in ['current', 'candidate', 'ort']},
        candidate_over_ort=corpus['ratios_to_ort']['candidate'])))


if __name__ == '__main__': main()
