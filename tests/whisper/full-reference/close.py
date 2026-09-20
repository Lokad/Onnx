"""Render all corpus comparisons and close only the complete 84-call campaign."""
import argparse,datetime
from common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',required=True);a=p.parse_args();base=Path(a.artifact).resolve()
    spec=read(base/'manifest.json');audit=read(base/'full-audit.json');assert audit['structural_passed'] and audit['jobs']==84 and audit['arrays']==3444
    assert audit['manifest']==pin(base/'manifest.json');verify(spec['files'])
    for name,wanted in audit['files'].items():assert pin(base/name)==wanted,name
    births=[]
    for phase in ['bridge','remaining']:
        state=read(base/(phase+'.json'));births.append(state['supervisor']);births += [r['worker'] for r in state['runs']]
    assert all(absent(i) for i in births)
    maximum=max(r['max_scaled'] for r in audit['reference_comparisons']);failed=sum(r['failed_values'] for r in audit['reference_comparisons'])
    lines=['# Complete Whisper encoder reference diagnostic — September 20, 2026','',
        f"All 84 reference calls, 3,444 boundary arrays and {audit['bytes']:,} output bytes are retained. The two float64 routes have maximum scaled difference **{maximum:.9g}**, with **{failed} values above 1e-9**. Reference agreement: **{'PASS' if audit['reference_passed'] else 'FAIL'}**.",'',
        'Each reference uses the same original FP32 weights/constants promoted exactly and the same saved features. One executes the original graph with NumPy/SciPy; the other uses ORT float64 graph sections, explicit convolution-to-matrix lowering and Python double math.erf between sections. ORT optimizations are disabled, with one intra/inter-op thread and no spinning. Neither route executes an FP32 Erf.','',
        '| Engine compared | Float64 reference | Failed final arrays / 42 | Failed values | Maximum scaled error |',
        '|---|---|---:|---:|---:|']
    for engine in ['managed','native']:
        for reference in ['numpy','ort']:
            rows=[r for r in audit['fp32_comparisons'] if r['engine']==engine and r['reference']==reference];assert len(rows)==42
            lines.append(f"| {engine} FP32 | {reference} | {sum(r['failed_values']>0 for r in rows)} | {sum(r['failed_values'] for r in rows):,} | {max(r['max_scaled'] for r in rows):.9g} |")
    lines+=['',
        'These use `abs(FP32-reference)/max(1,abs(reference)) <= 1e-4`, separately for both reference routes. All twenty original clips, both feature sources, the first-clip repeat and every padded frame are included. This does not replace the earlier direct managed/native gate, establish arbitrary-precision truth, or qualify a new product revision.','',
        '[Complete observations](observations-20260920.json) contain every request/boundary comparison, each FP32-versus-reference result, absolute/scaled maxima and coordinates, RMS/L2, every failed-value count, resource records and the complete file inventory. All repeated reference arrays retain identical bytes. The first two reference calls passed the prospectively fixed implementation bridge and were reused in the full inventory.','',
        f"The source revision is `{spec['source_revision']}`; original inference core SHA256 is `{spec['original_core']}`. Manifest SHA256 `{audit['manifest']['sha256']}`. The old results come from the [complete crossed-input corpus](../input-cross-isolated/results-20260920.md). No original FP32 inference was rerun, and no product arithmetic, default, model weight or tolerance was changed.",'',
        'All fresh workers ran sequentially on Windows logical CPU2, under a CPU0 supervisor. Each had 900 seconds / 4 GiB sampled RSS / 1 GiB minimum available memory, with 6 GiB available before launch. Original process identities are terminal. Resource samples cannot establish an absolute peak, and these diagnostic durations are not application latency measurements.','']
    folder=Path(__file__).parent;report=folder/'results-20260920.md';observations=folder/'observations-20260920.json'
    with report.open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines))
    write(observations,audit)
    receipt=dict(structural_passed=True,reference_passed=audit['reference_passed'],closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),births=births,
        reports={rel(p):pin(p) for p in [report,observations]},files={p.relative_to(base).as_posix():pin(p) for p in sorted(base.rglob('*')) if p.is_file()})
    write(base/'closed.json',receipt);print(json.dumps(dict(receipt=pin(base/'closed.json'),files=len(receipt['files']),reference_passed=audit['reference_passed'])))

if __name__=='__main__':main()
