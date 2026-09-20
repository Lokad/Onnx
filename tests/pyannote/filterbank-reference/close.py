"""Close complete evidence; numerical failures are reported without gate changes."""
import argparse, datetime
from common import *

def main():
    p = argparse.ArgumentParser(); p.add_argument('--artifact', required=True); a = p.parse_args()
    base = Path(a.artifact).resolve(); audit = read(base / 'audit.json'); spec = read(base / 'manifest.json')
    assert audit['structural_passed'] and audit['cases'] == 21 and audit['arrays'] == 294
    assert audit['manifest'] == pin(base / 'manifest.json') and all(absent(i) for i in audit['births'])
    verify(spec['files'])
    for name, wanted in audit['files'].items(): assert pin(base / name) == wanted, name
    lines = ['# Complete WeSpeaker filterbank reference — September 20, 2026', '',
             f"Both independent double-precision routes completed all **21 cases and 294 stage arrays** ({audit['bytes']:,} numeric bytes). Complete reference agreement: **{'PASS' if audit['reference_passed'] else 'FAIL'}**.", '',
             f"Maximum scaled stage difference is **{max(r['max_scaled'] for r in audit['stages']):.9g}**. Independent scalar preprocessing and direct Fourier sums check all 257 bins of each selected first/middle/last frame; maximum scaled difference is **{max(r['max_scaled'] for r in audit['scalars']):.9g}**. Both checks retain the prospective `1e-8` limit.", '',
             'The reference function uses the original PCM and saved native FP32 Hamming/mel coefficients, plus exactly promoted `.97f`, with all subsequent arithmetic in double. NumPy FFT/OpenBLAS and Torch double FFT/MKL use separately structured preprocessing, transforms, reductions, log and centering. This is a fixed-coefficient reference, not an arbitrary-precision proof or an ideal-coefficient replacement. Managed coefficient rounding is included in its observed discrepancy.', '',
             '| Original FP32 engine | Double reference | Failed arrays / 21 | Failed values / 711,680 | Maximum scaled error |',
             '|---|---|---:|---:|---:|']
    for engine in ['managed', 'native']:
        for reference in ['numpy', 'torch']:
            rows = [r for r in audit['fp32'] if r['engine'] == engine and r['reference'] == reference]
            assert len(rows) == 21
            lines.append(f"| {engine} | {reference} | {sum(r['failed'] > 0 for r in rows)} | {sum(r['failed'] for r in rows):,} | {max(r['max_scaled'] for r in rows):.9g} |")
    lines += ['', 'Every original FP32 array is reused and checked in full against both references under `abs(actual-reference)/max(1,abs(reference)) <= 1e-4`. The original direct managed/native comparison still has exactly **three failed French values**, maximum **1.258179657e-4**. This diagnostic does not change that gate or qualify full diarization, the separate segmentation silence discrepancy, or a new product revision.', '',
              '[Complete observations](observations-20260920.json) retain every stage, scalar check and FP32 comparison, including maxima/coordinates, RMS, failure counts and resource summaries. Original source is `98f075d`; the current frontend source is verified unchanged. [Protocol and commands](README.md) explain the fixed scope and independent implementations.', '',
              f"Tool source `{spec['source']}`; manifest SHA256 `{audit['manifest']['sha256']}`. Each fresh worker ran once on Windows CPU 0 with one numerical thread, under 180-second / 2 GiB sampled RSS / 1 GiB available-memory limits. All original worker and supervisor identities are terminal. Full inputs/coefficients, original outputs, source/library identities and every resource sample are retained. Diagnostic durations are not application latency measurements.", '']
    folder = Path(__file__).parent; report = folder / 'results-20260920.md'; observations = folder / 'observations-20260920.json'
    with report.open('x', encoding='utf-8') as stream: stream.write('\n'.join(lines))
    write(observations, audit)
    receipt = dict(structural_passed=True, reference_passed=audit['reference_passed'], closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   births=audit['births'], reports={rel(path): pin(path) for path in [report, observations]},
                   files={path.relative_to(base).as_posix(): pin(path) for path in sorted(base.rglob('*')) if path.is_file()})
    write(base / 'closed.json', receipt)
    print(json.dumps(dict(receipt=pin(base / 'closed.json'), files=len(receipt['files']), reference_passed=audit['reference_passed'])))

if __name__ == '__main__': main()
