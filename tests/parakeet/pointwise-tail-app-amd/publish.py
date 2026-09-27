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
        root_product_changed=False, component_screen_admitted=False, source=pin(Path(__file__)))
    target = ROOT/'tests/parakeet/decoder-lstm-layout-profile-results/pointwise-tail-app-observations-20260927.json'
    with target.open('x', encoding='utf8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(published=pin(target), application_admitted=performance['admitted'],
        controls_passed=sum(r['passed'] for r in performance['controls']),
        gates_passed=sum(r['passed'] for r in performance['gates']),
        corpus={role:corpus[role]['seconds'] for role in ['current', 'candidate', 'ort']},
        candidate_over_ort=corpus['ratios_to_ort']['candidate'])))


if __name__ == '__main__': main()
