"""Publish the closed complete-application comparison without changing its verdict."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'artifacts/parakeet-packed-final-row-app-amd-20260925'
OUT = Path(__file__).resolve().parent


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def main():
    proof = read(BASE / 'closed.json')
    value = read(BASE / 'analysis.json')
    assert proof['passed'] and value['passed'] and proof['analysis'] == pin(BASE / 'analysis.json')
    for name, wanted in proof['files'].items():
        assert pin(BASE / name) == wanted, name
    performance = value['performance']
    assert proof['admitted'] == performance['admitted']
    assert len(performance['controls']) == 63 and len(performance['gates']) == 21
    assert (value['timing_requests'], value['warmup'], value['measured']) == (480, 120, 360)
    assert not value['root_product_changed'] and value['separate_model_qualification_required']
    payload = read(BASE / 'payload.json')
    assert not payload['release_admitted'] and len(payload['failed_release_controls']) == 2
    assert payload['identities']['candidate']['Lokad.Onnx.dll']['sha256']=='49901366484570493b7a42028e5c5458f30fe2c1703a01335b68d9a5f9a1fea9'
    assert payload['identities']['candidate']['Lokad.Onnx.Data.dll']['sha256']=='01e9e7842f5e9861de3d6dc737db947c8a38f1a07038b403d5482ec676e810f1'
    corpus, = [row for row in value['table'] if row['is_corpus']]
    assert corpus['audio_seconds'] == 213.265 and len(value['table']) == 21
    gain = 1 - corpus['candidate']['seconds'] / corpus['current']['seconds']
    rows = [dict(name=row['name'], audio_seconds=row['audio_seconds'],
                 selected_seconds=row['current']['seconds'], candidate_seconds=row['candidate']['seconds'],
                 ort_seconds=row['ort']['seconds'], candidate_over_ort=row['ratios_to_ort']['candidate'],
                 gain=1-row['candidate']['seconds']/row['current']['seconds']) for row in value['table']]
    paths = [OUT / ('application-20260925' + suffix) for suffix in ['.json', '.csv', '.md']]
    assert not any(path.exists() for path in paths), 'Preserve existing publication'
    controls = sum(row['passed'] for row in performance['controls'])
    gates = sum(row['passed'] for row in performance['gates'])
    verdict = 'passes' if performance['admitted'] else 'fails'
    failed = [f"- Repeatability: {row['name']} / {row['role']}, {row['process_ratio']:.6f} > {row['limit']:.2f}."
              for row in performance['controls'] if not row['passed']]
    failed += [f"- Performance: {row['name']}, candidate/selected {row['candidate_over_current']:.6f} > {row['limit']:.2f}."
               for row in performance['gates'] if not row['passed']]
    failures = '\n\nFailed controls or gates:\n\n' + '\n'.join(failed) if failed else ''
    prose = f'''# Packed final row: complete Parakeet comparison

**The candidate {verdict} the unchanged application admission.** Complete
transcription latency is {100*abs(gain):.3f}% {'lower' if gain >= 0 else 'higher'} than the isolated M73 baseline.
Repeatability controls: {controls}/63. Performance gates: {gates}/21.
Candidate latency is {corpus['ratios_to_ort']['candidate']:.6f} times Microsoft ORT.
The separate 1.05 parity target is {'met' if performance['parity_target_met'] else 'not met'}.{failures}

| Twenty-clip corpus (213.265 s audio) | Seconds | Relative to ORT |
|---|---:|---:|
| Isolated M73 baseline | {corpus['current']['seconds']:.6f} | {corpus['ratios_to_ort']['current']:.6f} |
| M78 packed final-row candidate | {corpus['candidate']['seconds']:.6f} | {corpus['ratios_to_ort']['candidate']:.6f} |
| Microsoft ORT 1.29.0 | {corpus['ort']['seconds']:.6f} | 1.000000 |

Six fresh processes execute selected, candidate, ORT, ORT, candidate, selected.
Each runs one warmup and three measured passes over all 20 clips: 480 requests,
120 warmups and 360 measurements. Every clock contributes with equal process
weights. Ordinary application consumers run without profiling or runtime
overrides. Numerical, public-result, immutable-input and held-output checks pass.

The fixed thresholds require at least 3% corpus gain, no clip more than 5%
slower, corpus process repeatability at most 1.10 and per-clip at most 1.20
for each engine. Every process is terminal with code 0; all
{sum(row['samples'] for row in value['resources']):,} resource observations pass.
Peak owned RSS is {max(row['peak_rss'] for row in value['resources']):,} bytes.

This evaluates the complete owned-weight candidate with the targeted
remaining-row correction identified from the installed ORT path. It retains
87 constant weights in packed form and computes the remaining row directly
from that storage. Existing two-/three-row arithmetic, the nine already prepared
feed-forward weights and the clone budget are unchanged.
[Complete correctness](models-20260925.md) passes in both instruction modes;
[actual counters](counters-20260925.md) prove zero reconstruction events and
1,740 avoided transient packs per corpus. These counts are not independent
application savings. This fresh comparison measures the combined candidate
against M73; it does not isolate an incremental gain against historical M76.

The selected baseline is isolated M73, not the qualified repository release.
The prior e5-8tok and e5-512tok repeatability failures remain release blockers.
This result changes neither product source nor BENCHMARK.md. Shared models,
Pyannote, root/package and full-suite admission remain required before release.
A failed application admission retains its verdict and permits no unchanged retry.

[Every case](application-20260925.csv) and
[all controls, gates, identities, process clocks and resources](application-20260925.json)
are retained, including the two prior failed release controls.

Closure: `{pin(BASE / 'closed.json')['sha256']}`.
Raw evidence: `{BASE.relative_to(ROOT).as_posix()}`.
'''
    with paths[0].open('x', encoding='utf8') as stream:
        json.dump(dict(closure=pin(BASE / 'closed.json'), **value, release_admitted=False,
                       failed_release_controls=payload['failed_release_controls']), stream, indent=2, allow_nan=False)
    with paths[1].open('x', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with paths[2].open('x', encoding='utf8', newline='\n') as stream:
        stream.write(prose)
    print(json.dumps(dict(admitted=performance['admitted'], gain=gain,
                          candidate_over_ort=corpus['ratios_to_ort']['candidate'],
                          controls=controls, gates=gates, release_admitted=False)))


if __name__ == '__main__':
    main()
