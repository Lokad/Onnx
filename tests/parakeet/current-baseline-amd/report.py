"""Publish a closed baseline's complete table, clocks and setup intervals."""
import csv
import json
from pathlib import Path
from protocol import TIMING_ROLES, pin, read
from checks import evaluate
from run import BASE


def main():
    proof = read(BASE/'closed.json'); assert proof['passed']
    for name, wanted in proof['files'].items(): assert pin(BASE/name) == wanted, name
    value = read(BASE/'analysis.json')
    assert pin(BASE/'analysis.json') == proof['analysis'] and value['passed']
    assert evaluate(value['table']) == value['performance']
    out = Path(__file__).resolve().parent
    assert not (out/'results-20260922.md').exists()
    clocks, setups = [], []
    for process, role in enumerate(TIMING_ROLES):
        result = read(BASE/'collected'/f'timing-{process:02}-{role}'/'output/result.json')
        setups.append(dict(process=process, role=role, seconds=result['setup_seconds']))
        for row in result['records']:
            clocks.append(dict(process=process, role=role, **{key:row[key] for key in
                ['name','pass','phase','start_ticks','end_ticks','frequency','seconds']}))
    assert len(clocks) == 320 and sum(r['phase']=='measured' for r in clocks) == 240
    for filename, rows in [('clocks-20260922.csv',clocks), ('setup-20260922.csv',setups)]:
        with (out/filename).open('x', encoding='utf8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n'); writer.writeheader(); writer.writerows(rows)
    table = value['table']; corpus, = [r for r in table if r['is_corpus']]
    valid = value['performance']['baseline_valid']; current = value['identities']['current']
    lines = [
        '# Current Parakeet versus Microsoft ORT on AMD', '',
        f"The integrated M22 product completes the twenty-clip corpus in **{corpus['current']['seconds']:.6f} s**",
        f"versus **Microsoft ORT {corpus['ort']['seconds']:.6f} s**, ratio **{corpus['ratios_to_ort']['current']:.6f}**.",
        ('All 42 repeatability controls pass; this is a valid current baseline.' if valid else
         '**The baseline is invalid:** one or more fixed repeatability controls fail. All samples are retained; no unchanged retry.'),
        ('The separate <=1.05 application parity target is met on this corpus.' if value['performance']['parity_target_met'] else
         'The separate <=1.05 application parity target remains unmet.'), '',
        '| Clip | Audio seconds | Lokad.Onnx seconds | Microsoft ORT seconds | Lokad / ORT |',
        '|---|---:|---:|---:|---:|']
    for row in table:
        label = '**Complete corpus**' if row['is_corpus'] else row['name']
        lines.append(f"| {label} | {row['audio_seconds']:.3f} | {row['current']['seconds']:.9f} | {row['ort']['seconds']:.9f} | {row['ratios_to_ort']['current']:.6f} |")
    lines += ['', 'AMD EPYC 9V74, CPU2 before runtime startup, monitor CPU0; .NET 10.0.8 /',
        'SDK 10.0.204 and ORT 1.29.0. ORT uses CPUExecutionProvider, one intra/inter-op',
        'thread, sequential execution, all graph optimizations and no spinning.',
        'Native loaded libraries and all 20,661 external dependencies are pinned.',
        'Neither role enables a profiler or numerical overrides.', '',
        'Four fresh processes run current, ORT, ORT, current. Each performs one',
        'warmup and three measured passes over all twenty clips, totaling 213.265',
        'seconds of audio: **320 requests, 80 warmups and 240 measurements**.',
        'Each clip mean retains six measured calls across two equally weighted',
        'processes. Corpus means sum all twenty clip means within each process,',
        'then average both processes equally, using exact integer-clock fractions.',
        'No earlier campaign contributes a sample and no measurement is dropped.', '',
        'The timer includes frontend, neural inference, greedy decoding and owned',
        'public results. Model loading/setup, file access and external validation',
        'are separate. [Every raw clock](clocks-20260922.csv) and',
        '[all setup intervals](setup-20260922.csv) are retained.', '',
        'Repeatability requires process-mean max/min <=1.10 for the corpus and',
        '<=1.20 for each clip, separately for both roles. The complete decisions',
        'and process means are retained in [the observations](observations-20260922.json).',
        'This refresh introduces no product change and therefore has no',
        'candidate-versus-current improvement gate. It does not establish an',
        'improvement over older product timings from a separate campaign.', '',
        'Every request passes the original native/public transcript, token,',
        'duration, readonly-input and held-output checks. All 160 managed public',
        'results exactly match the retained M22 selected-product reference.',
        'Complete Parakeet tensors (784 arrays / 3,090,494 values), Pyannote,',
        'shared-model/e5, native conformance, long meetings and normal root/package',
        'qualification were already closed for these exact binaries. Their',
        'complete prerequisite identities and analyses are independently checked;',
        'those successful campaigns are not repeated here.', '',
        f"All four jobs end with code zero and every recorded owner is terminal. All **{sum(r['samples'] for r in value['resources']):,} resource observations** pass;",
        f"peak owned RSS is **{max(r['peak_rss'] for r in value['resources']):,} bytes**.",
        'Foreign-CPU snapshot checks pass, with their retained limitation that',
        'snapshots can miss CPU used by short-lived processes. The mandatory',
        'repeatability controls provide a separate check.', '',
        'Current measured Core:', '`'+current['Lokad.Onnx.dll']['sha256']+'`.',
        'Current measured Data:', '`'+current['Lokad.Onnx.Data.dll']['sha256']+'`.',
        'Unchanged AudioBenchmark:', '`'+value['consumers']['AudioBenchmark']['sha256']+'`.',
        'Selected source: `fe4eb657`; normal root build `b4f82542` / `cfa7e140`',
        'matches all 3,163 Core / 697 Data methods and public declarations.',
        'The measured binaries above supply this timing result; the rebuild',
        'equivalence does not create a separate rebuild performance claim.', '',
        'Artifact: `artifacts/parakeet-current-baseline-amd-20260922`.',
        'Closure SHA-256:', '`'+pin(BASE/'closed.json')['sha256']+'`.',
        'Payload SHA-256:', '`'+pin(BASE/'payload.json')['sha256']+'`.', '']
    (out/'results-20260922.md').write_text('\n'.join(lines), encoding='utf8', newline='\n')
    observations = dict(closed=pin(BASE/'closed.json'), analysis=pin(BASE/'analysis.json'),
        identities=value['identities'], table=table, performance=value['performance'], resources=value['resources'],
        clocks=pin(out/'clocks-20260922.csv'), setup=pin(out/'setup-20260922.csv'), generator=pin(Path(__file__)))
    (out/'observations-20260922.json').write_text(json.dumps(observations,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(baseline_valid=valid, corpus=corpus, report=pin(out/'results-20260922.md'))))


if __name__ == '__main__': main()
