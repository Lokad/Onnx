"""Close successfully audited diagnostic evidence without implying numerical acceptance."""
from pathlib import Path
import argparse,datetime,sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"input-cross"))
from common import ROOT,pin,read,write,verify
from audit import absent

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--artifact',type=Path,required=True);args=parser.parse_args();base=args.artifact.resolve()
    assert not (base/'closed.json').exists();spec=read(base/'manifest.json');verify(spec);audit=read(base/'audit.json')
    assert spec['protocol']=='whisper-input-cross-isolated-case-v3' and audit['workers']==42
    assert audit['manifest']==pin(base/'manifest.json') and audit['structural_passed'] is True and audit['arrays']==84 and audit['baseline_bridges']==42
    for name,want in audit['files'].items():assert pin(base/name)==want,name
    for resource in audit['resources']:assert all(absent(i) for i in resource['births'])
    report=Path(__file__).with_name('results-20260920.md');assert not report.exists()
    labels={'original_MM-NN':'Original pipelines: MM − NN','engine_MM-NM':'Same managed features: MM − NM',
            'engine_MN-NN':'Same native features: MN − NN','input_MM-MN':'Managed encoder input effect: MM − MN',
            'input_NM-NN':'Native encoder input effect: NM − NN','interaction':'Difference between the two input effects'}
    lines=['# Whisper encoder input and engine comparison — 2026-09-20','',
      'This diagnostic uses the same twenty clean-English recordings and final first-case repeat as the closed numerical replay. '
      'It holds model revision `360ebcde2559d60bb474678be3c1de9ef347d01a` and managed core `c6bf781` fixed. '
      'Every one of the 42 managed/native baseline bridges reproduces its saved output bytes exactly; all 84 newly saved encoder outputs pass structural, input, held-output and repeat checks.','',
      'The full corpus runs in 42 sequential fresh workers, one per engine and recording. Each makes its baseline call then its crossed-input call and terminates. Managed uses fresh Memory contexts per call. The actual baseline output is held through the second call and managed resets; request 20 is compared with request 0 across processes for all four combinations. This does not test a long-lived public-transcriber process. The [fresh-context attempt](../input-cross/resource-failure-20260920.md) and [reused-context attempt](../input-cross-reuse/resource-failure-20260920.md) remain separately closed resource failures.','',
      'MM is the managed encoder on managed features; NN is the native encoder on native features. '
      'MN is the managed encoder on native features; NM is the native encoder on managed features. '
      'The original feature arrays are reused directly. No frontend, decoder, transcription or performance comparison is rerun.','',
      '| Difference | Failed arrays / 21 | Failed values / 40,320,000 | Maximum absolute | Maximum scaled | L2 norm |',
      '|---|---:|---:|---:|---:|---:|']
    for term,row in audit['aggregate'].items():
        lines.append(f"| {labels[term]} | {row['failed_arrays']} | {row['failed_values']:,} | {row['max_abs']:.9g} | {row['max_scaled']:.9g} | {row['l2']:.9g} |")
    lines+=['','Every scaled difference uses the original native denominator `max(1, abs(NN))`; the unchanged threshold is `1e-4`. '
      'The input-effect and interaction rows describe sensitivity, not alternative acceptance tests. '
      'The first recording repeat is included, matching the prior numerical scope. '
      'Both algebraic paths reconstruct the original difference within `1e-12` in float64. '
      'L2 norms include all saved values; opposing terms may cancel, so component norms are not causal percentages.','',
      'These observations do not identify which FP32 engine is closer to mathematical truth. '
      'An independent higher-precision reference is still needed for that question. '
      'The original pipeline failures remain failures regardless of a crossed-input result.','',
      'All 42 workers run sequentially on Windows i7-14700KF, logical CPU 2. Managed uses .NET 10.0.12 and the exact prior core SHA '
      '`7653c1686419d612e2624740908a44ffdee91b4239628a979bac44f5b6b863e9`. '
      'Native uses the original Python environment, ORT 1.29.0 and NumPy 2.2.4, one intra/inter-op thread, sequential execution, all graph optimizations and no spinning. '
      'The new manifest pins the actual ORT Python/native-library bytes; loaded native modules are checked. '
      'The older reference receipt had package versions but no library hashes, which is why all saved baseline cases must bridge exactly.','',
      'Workers load only the encoder. Guards require 10 GiB available before launch, at most 8 GiB sampled process-group RSS, '
      'at least 1 GiB available while running and at most 1,800 seconds per worker. '
      'Process identities include PID and creation time; all observed births are terminal. '
      'Sampling does not prove an absolute peak or observe children that begin and end between samples.','']
    for engine in ['managed','native']:
        values=[r for r in audit['resources'] if r['job']['engine']==engine]
        lines.append(f"- {engine}: 21 workers; {sum(r['samples'] for r in values):,} resource samples; peak group RSS {max(r['peak_rss'] for r in values):,} bytes; minimum available {min(r['minimum_available'] for r in values):,} bytes.")
    lines+=['',f"Manifest SHA-256: `{pin(base/'manifest.json')['sha256']}`. Full audit SHA-256: `{pin(base/'audit.json')['sha256']}`.",
      'Full arrays, observations and receipts are retained in `artifacts/whisper-input-cross-isolated-20260920`; source and refusal tests are in this directory. '
      'This is a localization experiment on the original qualified core, not current-product or general model qualification.','']
    report.write_text('\n'.join(lines),encoding='utf-8')
    files={p.relative_to(base).as_posix():pin(p) for p in sorted(base.rglob('*')) if p.is_file()}
    write(base/'closed.json',dict(schema=1,closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),structural_passed=True,
        numerical_original_passed=audit['aggregate']['original_MM-NN']['failed_arrays']==0,files=files,
        report=dict(file=report.relative_to(ROOT).as_posix(),**pin(report))))
    print('Closed',len(files),'files;',pin(base/'closed.json'))

if __name__=='__main__':main()
