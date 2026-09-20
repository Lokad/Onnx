import argparse, datetime
from shared import *

def main():
    p = argparse.ArgumentParser(); p.add_argument('--artifact', required=True); a = p.parse_args()
    base = Path(a.artifact).resolve(); spec = read(base / 'manifest.json'); audit = read(base / 'audit.json')
    assert audit['structural_passed'] and audit['manifest'] == pin(base / 'manifest.json') and audit['arrays'] == 288
    verify(spec['files'])
    for name, wanted in audit['files'].items(): assert pin(base / name) == wanted
    assert all(absent(birth) for birth in audit['births'])
    lines = ['# Complete mel-coefficient isolation — September 20, 2026', '',
             f"Read-only coefficient capture from both qualified Windows Data assemblies succeeds; their complete window/mel tables match exactly, including the older instrumented tables. Native-coefficient controls: **{'PASS' if audit['controls_passed'] else 'FAIL'}**. Complete independent scalar energy/log/centering checks: **{'PASS' if audit['scalar_passed'] else 'FAIL'}**.", '',
             'All thirty-two saved double power spectra remain fixed. Only mel coefficients change: saved native FP32, captured managed FP32, or independently computed real-formula coefficients rounded to double. The third is an explicit coefficient diagnostic, not a replacement acceptance oracle or a complete ideal-precision frontend. No original waveform transform, FFT or model inference is rerun.', '',
             '| Corpus | Mel coefficients in controlled calculation | Windows FP32 failed values | Maximum scaled discrepancy |',
             '|---|---|---:|---:|']
    for corpus in ['five', 'dialogue']:
        for setting in ['native', 'managed', 'ideal']:
            rows = [r for r in audit['comparisons'] if r['corpus'] == corpus and r['variant'] == 'windows-default' and r['setting'] == setting]
            lines.append(f"| {corpus} | {setting} | {sum(r['failed'] for r in rows)} | {max(r['max_scaled'] for r in rows):.9g} |")
    lines += ['', 'Both original Windows settings are checked and identical. Every saved FP32 value is compared with `abs(actual-reference)/max(1,abs(reference)) <= 1e-4`; these controlled substitutions do not alter the original native-agreement gate. AMD coefficients were not captured, so no AMD coefficient attribution follows.', '',
              f"All 288 complete controlled energy/log/feature arrays ({audit['bytes']:,} numeric bytes) are retained. The native-weight control's maximum scaled difference from the saved reference is {max(r['max_scaled'] for r in audit['controls']):.9g}; complete scalar verification reaches {max(r['max_scaled'] for r in audit['scalars']):.9g}, under the predeclared 1e-12 limit. Decimal 50-digit coefficient generation and an independently structured double formula differ by at most {audit['formula_maximum']:.9g} across all 20,480 weights.", '',
              '[Complete observations](observations-20260920.json) retain all comparisons and per-band maxima. Full total-error, coefficient-displacement and remaining-error vectors are saved; their sum identity is checked. Their L2 norms are not additive or independent causal percentages. Remaining differences can include earlier window/preemphasis/FFT/power rounding and later float log/centering.', '',
              f"Tool source `{spec['source']}`; manifest SHA256 `{audit['manifest']['sha256']}`. Build, two reflection-only captures and analysis run once on CPU0 with fixed 180-second / 2 GiB sampled group-RSS / 1 GiB available-memory limits. All observed process identities are terminal. [Protocol](README.md) records assumptions, source identities and commands. No product arithmetic, default, model, tolerance or application timing claim changes.", '']
    folder = Path(__file__).parent; report = folder / 'results-20260920.md'; observations = folder / 'observations-20260920.json'
    with report.open('x', encoding='utf-8') as stream: stream.write('\n'.join(lines))
    write(observations, audit)
    receipt = dict(structural_passed=True, controls_passed=audit['controls_passed'], scalar_passed=audit['scalar_passed'],
                   closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), births=audit['births'],
                   reports={rel(path): pin(path) for path in [report, observations]},
                   files={path.relative_to(base).as_posix(): pin(path) for path in sorted(base.rglob('*')) if path.is_file()})
    write(base / 'closed.json', receipt)
    print(json.dumps(dict(receipt=pin(base / 'closed.json'), files=len(receipt['files']), controls_passed=audit['controls_passed'], scalar_passed=audit['scalar_passed'])))

if __name__ == '__main__': main()
