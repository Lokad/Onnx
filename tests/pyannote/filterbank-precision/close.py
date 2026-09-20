"""Close the complete precision experiment, retaining original native failures."""
import argparse, datetime
from shared import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',required=True);a=p.parse_args();base=Path(a.artifact).resolve()
    spec=read(base/'manifest.json');audit=read(base/'audit.json');extra=read(base/'compatibility.json')
    assert audit['structural_passed'] and audit['diagnostic_passed'] and audit['manifest']==pin(base/'manifest.json')
    assert audit['arrays']==2862 and extra['complete'] and extra['audit']==pin(base/'audit.json')
    verify(spec['files'])
    for name,wanted in audit['files'].items():assert pin(base/name)==wanted,name
    assert extra['source']==pin(Path(__file__).with_name('compatibility.py'))
    for row in extra['records']:assert row['gaps']==pin(base/'compatibility'/(row['name']+'.npy'))
    births=audit['births']+[audit['auditor'],extra['process']];assert all(absent(b) for b in births)
    def row(variant,reference):return next(r for r in audit['summary'] if (r['variant'],r['corpus'],r['reference'])==(variant,'all',reference))
    maximum=max(r['max_scaled'] for r in audit['references'])
    lines=['# WeSpeaker intermediate-precision experiment — September 20, 2026','',
        '**Frame preprocessing is the nominated arithmetic change.** Keeping frame mean, DC subtraction, preemphasis and window application in double removes every feature-gate failure against both independent fixed-coefficient double references across all53 cases. Widening only the spectrum or output group does not. Product code and numerical acceptance remain unchanged.','',
        'All variants keep the captured managed float Hamming/mel coefficients, exactly promoted `.97f`, original FFT, framing and final float output. The original generated implementation reproduces all53 saved Windows arrays bit for bit. The new managed-table NumPy/OpenBLAS and Torch/MKL references agree at every complete stage, maximum scaled difference '+f'{maximum:.9g}'+'. The existing native-table reference arrays remain separate comparisons; neither reference function was relabelled.','',
        '| Arithmetic | Failures vs managed-table NumPy | Maximum scaled error | Total squared error | Failures vs native float | Failures vs old native-table NumPy |',
        '|---|---:|---:|---:|---:|---:|']
    for variant in VARIANTS:
        r=row(variant,'numpy');native=row(variant,'native');old=row(variant,'old-numpy')
        lines.append(f"| {variant} | {r['failed']} | {r['max_scaled']:.9g} | {r['squared_error']:.9g} | {native['failed']} | {old['failed']} |")
    lines+=['',
        'Each row covers3,266,560 feature values. Torch gives the same failure counts for both coefficient policies; full metrics for each reference are retained. Frame reduces aggregate squared error against the managed-table NumPy reference by '+f"{100*(1-row('Frame','numpy')['squared_error']/row('Original','numpy')['squared_error']):.4f}%"+'. Its maximum error against the old native-table references is'+f" {max(row('Frame',r)['max_scaled'] for r in ['old-numpy','old-torch']):.9g}"+', also below the unchanged1e-4 gate. All-double is more accurate still, but changes three groups rather than one; the prospective rule prefers the smaller qualifying change.','',
        '**Direct native agreement does not improve:** Frame has187 failed values versus the original32, while native itself has185 failures against each managed-table double reference. These counts are preserved, not waived. No model, embedding, clustering or public diarization replay was performed, and no new product/default/native-conformance claim follows.','',
        'A separately identified post-hoc feasibility calculation intersects the unchanged1e-4 tolerance intervals around native and both new references for every value. '+f"**{extra['incompatible']} coordinates have disjoint intervals**"+', confirmed individually with90-digit Decimal arithmetic. Therefore no output can satisfy all three of those gates at those coordinates. Complete separation arrays and every conflicting coordinate are retained. This establishes an incompatibility among those fixed gates for these inputs; it does not itself replace the acceptance contract.','',
        'The53 cases comprise the existing21 frontend cases and32 pipeline windows: boundary lengths through480000 samples, noise, silence, DC, quiet, impulse, tones, recordings and the annotated dialogue. Three repeated input windows reproduce every saved stage exactly. These are not53 independent recordings. Inputs and coefficients remain unchanged; synthetic zero/centering checks pass for the nominated variant.','',
        f"All{audit['arrays']:,} complete stage/final-output arrays ({audit['numeric_bytes']:,} numeric bytes) are retained. Each of five managed variants saves seven complete stages plus actual float feature bytes; each double reference saves seven stages. Independent scalar window/direct Fourier checks cover every bin of fixed first/middle/last frames, maximum scaled error{max(r['max_scaled'] for r in audit['scalars']):.9g}. All-double stages pass the fixed1e-8 bound; its final float output uses the separately declared1e-6 comparison to rounded reference values. No observed threshold was changed.",'',
        '| Stage | Seconds | Resource samples | Peak process-group RSS bytes |',
        '|---|---:|---:|---:|']
    for r in audit['resources']:lines.append(f"| {r['name']} | {r['seconds']:.3f} | {r['samples']} | {r['peak_group_rss']} |")
    lines+=['',
        'Build and three computation workers each run once on Windows CPU0 under600second/2GiB group-RSS/1GiB available-memory guards. .NET10.0.12 build has zero warnings/errors. Numerical threads are one; NumPy2.2.4 and Torch2.11.0+cpu binary identities are bound. All observed computation, build-child, supervisor, auditor and feasibility-process identities are terminal. These diagnostic durations are not application latency.','',
        f"Frozen experiment source `{spec['source']}`, manifest `{audit['manifest']['sha256']}`. Artifact `artifacts/wespeaker-precision-20260920`. [Protocol and commands](README.md), [complete observations](observations-20260920.json), and [prior coefficient diagnostic](../filterbank-coefficients/results-20260920.md). Next work is a prospectively defined frontend acceptance contract and actual product/host/application qualification, retaining the direct-native discrepancies and all other model gates.",'']
    lane=Path(__file__).resolve().parent;report=lane/'results-20260920.md';data=lane/'observations-20260920.json'
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
    write(data,dict(audit=audit,compatibility=extra))
    receipt=dict(closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),diagnostic_passed=True,nominated=audit['nominated'],births=births,
        reports={rel(p):pin(p) for p in [report,data,Path(__file__),Path(__file__).with_name('compatibility.py')]},
        files={p.relative_to(base).as_posix():pin(p) for p in sorted(base.rglob('*')) if p.is_file()})
    write(base/'closed.json',receipt);print(json.dumps(dict(receipt=pin(base/'closed.json'),files=len(receipt['files']),nominated=audit['nominated'])))


if __name__=='__main__':main()
