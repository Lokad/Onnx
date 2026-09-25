"""Join the unchanged failed application verdict to prior per-clip copy counts."""
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
APP = ROOT / 'artifacts/parakeet-owned-packed-weight-app-amd-20260925'


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def exact(value):
    return Fraction(**value['exact_mean'])


def main():
    assert pin(APP / 'closed.json')['sha256'] == '959d8d180db3431e1aeb5d7ab555fd9b77035ae84d4a8a678d2204cccc71eb48'
    proof = read(APP / 'closed.json')
    assert proof['passed'] and not proof['admitted'] and proof['analysis'] == pin(APP / 'analysis.json')
    app = read(APP / 'analysis.json')
    counts_path = OUT / 'counters-20260925.json'
    assert pin(counts_path)['sha256'] == 'bec88d58724e9d1a5ea4f61cf913349888a5580c45a45f1b9542badca2977ca9'
    counts = read(counts_path)
    assert counts['passed'] and counts['diagnostic_only'] and not counts['performance_measured']
    assert counts['products'] == {'selected': app['identities']['current'], 'candidate': app['identities']['candidate']}
    performance = app['performance']
    assert performance['controls_passed'] and sum(r['passed'] for r in performance['controls']) == 63
    assert [r['name'] for r in performance['gates'] if not r['passed']] == ['corpus-at-least-three-percent-gain']
    normal, disabled = counts['comparisons']
    assert (normal['mode'], disabled['mode']) == ('512', '256')
    assert normal['clips'] == disabled['clips']
    by_name = {row['name']: row for row in normal['clips']}
    cases = [row for row in app['table'] if not row['is_corpus']]
    assert len(by_name) == len(cases) == 20 and set(by_name) == {r['name'] for r in cases}
    groups = []
    rows = []
    for reconstruct in [False, True]:
        selected = Fraction(0)
        candidate = Fraction(0)
        names = []
        slower = 0
        for row in cases:
            count = by_name[row['name']]
            expected = count['frames'] % 2 != 0 and count['frames'] % 3 != 0
            assert bool(count['reconstructions']) == expected
            assert count['avoided_packs'] == 87 and count['reconstructions'] == (87 if expected else 0)
            assert count['outputs_exact'] and count['copy_increase'] == (1459617792 if expected else 0)
            if expected != reconstruct:
                continue
            before = exact(row['current'])
            after = exact(row['candidate'])
            selected += before
            candidate += after
            names.append(row['name'])
            slower += after > before
            rows.append(dict(name=row['name'], frames=count['frames'], reconstructions=count['reconstructions'],
                             selected_seconds=float(before), candidate_seconds=float(after),
                             saved_seconds=float(before-after), gain=float(1-after/before)))
        groups.append(dict(reconstruction=reconstruct, cases=len(names), names=names, slower_cases=slower,
                           selected_seconds=float(selected), candidate_seconds=float(candidate),
                           saved_seconds=float(selected-candidate), gain=float(1-candidate/selected)))
    assert [g['cases'] for g in groups] == [13, 7]
    assert [g['slower_cases'] for g in groups] == [0, 6]
    corpus, = [r for r in app['table'] if r['is_corpus']]
    assert sum(exact(r['current'])-exact(r['candidate']) for r in cases) == exact(corpus['current'])-exact(corpus['candidate'])
    paths = [OUT / ('shortfall-20260925'+suffix) for suffix in ['.json', '.csv', '.md']]
    assert not any(p.exists() for p in paths)
    text = '''# Where the owned-weight application gain falls short

The unchanged complete-application admission fails: 2.590% gain against the
required 3%, despite all 63 repeatability controls and all 20 per-clip gates
passing. The failed verdict remains. This review partitions every clip using
the reconstruction condition established before timing; it creates no new score
or admission rule.

| Existing execution path | Clips | Selected seconds | Candidate seconds | Saved seconds | Latency change |
|---|---:|---:|---:|---:|---:|
'''
    for row in groups:
        label = 'No weight reconstruction' if not row['reconstruction'] else 'Dense weights reconstructed for final row'
        change = f"{abs(row['gain'])*100:.3f}% {'lower' if row['gain'] >= 0 else 'higher'}"
        text += f"| {label} | {row['cases']} | {row['selected_seconds']:.6f} | {row['candidate_seconds']:.6f} | {row['saved_seconds']:.6f} | {change} |\n"
    text += '''
All 13 clips without reconstruction improve. Six of the seven reconstruction
clips become slower; the seventh improves by only 0.247%. Those seven lengths
are 167, 89, 157, 83, 61, 151 and 169: odd values not divisible by three.
The actual counter audit recorded 609 additional 16 MiB reconstructions,
10,217,324,544 cumulative copy bytes, on exactly those inputs in both modes.

This alignment identifies a specific diagnostic target, not a causal timing
attribution. The groups contain different sequence lengths and original costs.
The 0.338-second group regression is not the cost of copying: it is the net
effect of saved packing, reconstruction and any cache/runtime changes.
Do not add the 1.875-second saving from the other group to an assumed copy cost.

The next bounded check is to retain the existing CopyY stage durations and
complete feed-forward call durations on these exact products and clips. The
existing counter consumer already observes those stages but discards their
times. Reuse the unchanged Core/Data binaries, preserve every clip and result,
and check repeatability before choosing a remedy. No cache-size sweep, new
arithmetic kernel or unchanged application retry is selected.

[Complete application result](application-20260925.md),
[prior actual packing/reconstruction counts](counters-20260925.md),
[every clip](shortfall-20260925.csv), [identities and group totals](shortfall-20260925.json).
The qualified product and BENCHMARK.md remain unchanged. No new model execution
was needed for this review.
'''
    with paths[0].open('x', encoding='utf8') as stream:
        json.dump(dict(application_closure=pin(APP / 'closed.json'), analysis=pin(APP / 'analysis.json'),
                       counters=pin(counts_path), groups=groups, clips=rows, application_admitted=False,
                       new_admission_rule=False, causal_attribution=False, reviewer=pin(Path(__file__))),
                  stream, indent=2, allow_nan=False)
    with paths[1].open('x', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with paths[2].open('x', encoding='utf8', newline='\n') as stream:
        stream.write(text)
    print(json.dumps(dict(passed=True, application_admitted=False, groups=groups), indent=2))


if __name__ == '__main__':
    main()
