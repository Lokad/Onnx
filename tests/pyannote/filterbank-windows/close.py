"""Report all prior corpus/host/setting comparisons and close immutable evidence."""
import argparse, datetime
from shared import *

def main():
    p = argparse.ArgumentParser(); p.add_argument('--artifact', required=True); a = p.parse_args()
    base = Path(a.artifact).resolve(); audit = read(base / 'audit.json'); spec = read(base / 'manifest.json')
    assert audit['structural_passed'] and audit['cases'] == 32 and audit['arrays'] == 448
    assert audit['manifest'] == pin(base / 'manifest.json') and all(absent(i) for i in audit['births'])
    verify(spec['files'])
    for name, wanted in audit['files'].items(): assert pin(base / name) == wanted
    lines = ['# Complete pipeline filterbank window references — September 20, 2026', '',
             f"Both independent double routes complete **32 windows / 448 complete stage arrays**, retaining {audit['bytes']:,} numeric bytes. Reference agreement: **{'PASS' if audit['reference_passed'] else 'FAIL'}**. Maximum scaled stage difference is **{max(r['max_scaled'] for r in audit['stages']):.9g}**, and independent scalar preprocessing/direct Fourier checks reach **{max(r['max_scaled'] for r in audit['scalars']):.9g}**, both below the prospective `1e-8` limit." if audit['reference_passed'] else 'Reference agreement failed; see every retained comparison.', '',
             'The 24 dialogue windows cover the full thirty-second annotated recording and its three ten-second crops. Eight earlier pipeline windows cover English, French, JFK and the two-recording sequence, including right padding. Those earlier recordings already occurred in the original frontend corpus; they are new window/centering conditions, not independent new audio. The crops duplicate full-dialogue windows 0/10/20; every repeated reference stage matches exactly.', '',
             'The same fixed-coefficient reference function and unchanged NumPy/Torch routes from the [original complete frontend reference](../filterbank-reference/results-20260920.md) are used. Saved native FP32 window/mel constants and `.97f` are promoted exactly; subsequent arithmetic is double. Original model/frontend inference is not rerun. This does not prove ideal-coefficient or arbitrary-precision accuracy.', '',
             '| Corpus | Saved FP32 variant | Reference | Failed arrays | Failed values | Maximum scaled error |',
             '|---|---|---|---:|---:|---:|']
    for corpus in CORPORA:
        for variant in VARIANTS:
            for reference in ['numpy', 'torch']:
                rows = [r for r in audit['comparisons'] if r['corpus'] == corpus['name'] and r['variant'] == variant and r['reference'] == reference]
                assert len(rows) == corpus['windows']
                lines.append(f"| {corpus['name']} | {variant} | {reference} | {sum(r['failed'] > 0 for r in rows)}/{len(rows)} | {sum(r['failed'] for r in rows):,} | {max(r['max_scaled'] for r in rows):.9g} |")
    lines += ['', 'Every comparison uses `abs(actual-reference)/max(1,abs(reference)) <= 1e-4`. All 2,554,880 values per variant are included, with both saved settings checked independently and confirmed identical within each host. The native arrays are the original Windows Torch/Torchaudio frontend reference shared by both host replays; no same-host AMD native frontend run is inferred.', '',
              'The original direct managed/native filterbank failures remain: ten values on each host in the earlier corpus, and nineteen Windows / twenty-four AMD values in the dialogue corpus, per setting. The old silence case has no filterbank and retains its separate 3,458 failed segmentation scores. No tolerance, product arithmetic, default, native-agreement requirement or application result is changed by this diagnostic.', '',
              '[Complete observations](observations-20260920.json) contain every original and wider-reference comparison, all failed-value counts, maxima/coordinates, RMS, scalar checks, duplicate identities and resource observations. Original product revisions are `94a4f7c` and `21f3e74`; the current frontend and window rules are verified unchanged. See the [protocol](README.md) for fixed scope and commands.', '',
              f"Tool source `{spec['source']}`; manifest SHA256 `{audit['manifest']['sha256']}`. Both workers run once on CPU 0 with one numerical thread; fixed limits are 180 seconds, 2 GiB sampled RSS and 1 GiB available memory, with 4 GiB available before launch. All worker/supervisor identities are terminal. Retained diagnostic times are not application latency measurements.", '']
    folder = Path(__file__).parent; report = folder / 'results-20260920.md'; observations = folder / 'observations-20260920.json'
    with report.open('x', encoding='utf-8') as stream: stream.write('\n'.join(lines))
    write(observations, audit)
    receipt = dict(structural_passed=True, reference_passed=audit['reference_passed'], closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), births=audit['births'],
                   reports={rel(path): pin(path) for path in [report, observations]},
                   files={path.relative_to(base).as_posix(): pin(path) for path in sorted(base.rglob('*')) if path.is_file()})
    write(base / 'closed.json', receipt)
    print(json.dumps(dict(receipt=pin(base / 'closed.json'), files=len(receipt['files']), reference_passed=audit['reference_passed'])))

if __name__ == '__main__': main()
