"""Recompute remaining e5 latency deficits without running new inference."""
from pathlib import Path
import datetime, hashlib, json, math, statistics, subprocess

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/e5-remaining-gap-review-v2-20260920'
FOLDER=Path(__file__).resolve().parent
CASES=['e5-8tok','e5-30tok','e5-30pad128','e5-128tok','e5-512tok']


def pin(path):
    with path.open('rb') as f:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


def main():
    assert not BASE.exists();BASE.mkdir();bindings={};verification=[]
    def bind(path,wanted=None):
        actual=pin(path)
        if wanted is not None:assert actual==wanted if isinstance(wanted,dict) else actual['sha256']==wanted,str(path)
        bindings[path.relative_to(ROOT).as_posix()]=actual
        return read(path)
    public_base=ROOT/'artifacts/e5-public-ort-20260919';public_receipt=bind(public_base/'receipt.json')
    public=bind(public_base/'summary.json',public_receipt['files']['summary.json'])
    assert public['execution_complete'] and public['measured_calls']==5940 and len(public['reports'])==90
    for visit in public['reports']:
        for boundary in ['execute','request']:
            row=visit[boundary];assert len(row['samples'])==33
            assert math.isclose(statistics.mean(row['samples']),row['mean'],rel_tol=1e-14)
    verification.append(dict(source='public isolated',raw_measured_calls=5940,processes=90))
    profile_base=ROOT/'artifacts/e5-current-attribution-20260919';profile_receipt=bind(profile_base/'receipt.json')
    profile=bind(profile_base/'summary.json',profile_receipt['files']['summary.json'])
    provenance=bind(profile_base/'provenance.json',profile_receipt['files']['provenance.json'])
    assert profile['execution_complete'] and profile['profile_calls']==150 and len(profile['reports'])==10
    for visit in profile['reports']:
        for row in [visit['profiled_execute'],visit['outside_nodes'],*visit['families'].values()]:
            assert len(row['samples'])==15 and math.isclose(statistics.mean(row['samples']),row['mean'],rel_tol=1e-14)
    verification.append(dict(source='historical managed attribution',profiled_calls=150,visits=10,flags=provenance['flags']))
    resident_base=ROOT/'artifacts/e5-interleaved-processes-v3-20260920';resident_receipt=bind(resident_base/'aa-closed.json')
    resident=bind(resident_base/'aa-audit.json',resident_receipt['files']['aa-audit.json'])
    assert resident_receipt['evidence_passed'] and not resident_receipt['timing_passed']
    assert resident['passed'] and not resident['timing']['passed'] and len(resident['timing']['cases'])==10
    gaps=[]
    for case in public['cases']:
        native=case['configurations']['ort']['execute']['mean']
        for policy in ['default','memory']:
            rows=[r['execute']['samples'] for r in public['reports'] if r['name']==case['name'] and r['config']==policy]
            native_rows=[r['execute']['samples'] for r in public['reports'] if r['name']==case['name'] and r['config']=='ort']
            assert len(rows)==len(native_rows)==6
            managed=statistics.mean(x for row in rows for x in row)
            assert math.isclose(managed,case['configurations'][policy]['execute']['mean'],rel_tol=1e-14)
            assert math.isclose(native,statistics.mean(x for row in native_rows for x in row),rel_tol=1e-14)
            gaps.append(dict(protocol='isolated',case=case['name'],policy=policy,managed_ms=managed,ort_ms=native))
    for case in resident['timing']['cases']:
        means=case['boundaries']['execute']['role_mean_seconds']
        gaps.append(dict(protocol='resident',case=case['case'],policy=case['policy'],managed_ms=1000*statistics.mean(means[r] for r in ['A','B','C']),ort_ms=1000*means['N']))
    for row in gaps:
        assert row['case'] in CASES and row['managed_ms']>0 and row['ort_ms']>0
        deficit=max(0.,row['managed_ms']-1.05*row['ort_ms'])
        row.update(ratio=row['managed_ms']/row['ort_ms'],target_ms=1.05*row['ort_ms'],deficit_ms=deficit,required_reduction_percent=100*deficit/row['managed_ms'],primary=row['case']!='e5-512tok')
    assert len(gaps)==20
    attribution=[]
    for case in CASES:
        rows=[r for r in profile['reports'] if r['name']==case];assert len(rows)==2
        for visit,row in enumerate(rows):
            families={k:v['mean'] for k,v in row['families'].items()}
            assert math.isclose(math.fsum(families.values())+row['outside_nodes']['mean'],row['profiled_execute']['mean'],rel_tol=1e-12)
            attribution.append(dict(case=case,visit=visit,profile_ms=row['profiled_execute']['mean'],families_ms=families,outside_nodes_ms=row['outside_nodes']['mean'],matmul_fraction=families['MatMul']/row['profiled_execute']['mean']))
    layer_base=ROOT/'artifacts/e5-layernorm-minimum-20260920';layer_receipt=bind(layer_base/'closed.json')
    layer_path=ROOT/'tests/e5/layernorm-minimum/observations-20260920.json'
    layer=bind(layer_path,layer_receipt['reports'][layer_path.as_posix()]);assert layer['overall_passed']
    normalization=[]
    for case in layer['cases']:
        if case['diagnostic']:continue
        means={name:statistics.mean(r['mean_ms'] for r in layer['rows'] if r['case']==case['name'] and r['variant']==name) for name in ['Product','Wide']}
        for name,mean in means.items():assert math.isclose(mean,case['means_ms'][name],rel_tol=1e-14)
        normalization.append(dict(case=case['name'],**means,reduction_ms=means['Product']-means['Wide'],reduction_percent=100*(1-means['Wide']/means['Product'])))
    cache_base=ROOT/'artifacts/e5-fingerprint-cache-v3-20260920';cache_receipt=bind(cache_base/'closed.json')
    cache=bind(ROOT/'tests/e5/fingerprint-cache/observations-20260920.json')
    assert pin(cache_base/'closed.json')==cache['receipt'] and cache_receipt['component_screen_passed']
    assert cache['audit']['timing']['passed'] and len(cache['workers'])==4
    for index,worker in enumerate(cache['workers']):
        name=f'collected/timing-process/v{index}/output/result.json'
        assert bind(cache_base/name,cache_receipt['files'][name])==worker
    cache_means=[]
    for variant in range(4):
        means=[]
        for worker in cache['workers']:
            rows=[r for r in worker['samples'] if r['variant']==variant];assert len(rows)==64
            means.append(statistics.mean(r['ticks']/worker['frequency']/worker['repeats'] for r in rows))
        cache_means.append(statistics.mean(means))
        assert math.isclose(cache_means[-1],cache['audit']['timing']['mean_seconds'][variant],rel_tol=1e-14)
    ort_root=ROOT/'external/onnxruntime'
    assert subprocess.check_output(['git','-C',str(ort_root),'rev-parse','HEAD'],text=True).strip()=='a83fc4d58cb48eb68890dd689f94f28288cf2278'
    source_names=['src/Lokad.Onnx/MathOps.PackedAvx512.cs','src/Lokad.Onnx/MathOps.PackedPanels.cs',
                  'src/Lokad.Onnx/TensorOps.Norm.cs','docs/runtime-options.md',
                  'external/onnxruntime/onnxruntime/core/mlas/lib/sgemm.cpp',
                  'external/onnxruntime/onnxruntime/core/mlas/lib/x86_64/FgemmKernelAvx512FCommon.h']
    report_names=['tests/e5/projection-reduction-timing/results-20260920.md','tests/e5/projection-input-pack/results-20260919.md',
                  'tests/e5/projection-input-pointer/results-20260919.md','tests/e5/packed-overwrite-20260919.md',
                  'tests/e5/gelu-uniform-amd/results-20260920.md','tests/e5/fingerprint-balanced/comparison-results-20260920.md',
                  'tests/e5/interleaved-processes/aa-results-20260920.md']
    for name in source_names+report_names:bindings[name]=pin(ROOT/name)
    for path in sorted((ROOT/'artifacts/e5-remaining-gap-review-20260920').iterdir()):
        if path.is_file():bindings[path.relative_to(ROOT).as_posix()]=pin(path)
    value=dict(scope='Local synthesis of closed evidence; no new inference, timing or universal performance bound',
        created=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        gaps=gaps,attribution=attribution,profile_source=profile['source'],profile_flags=provenance['flags'],normalization=normalization,
        fingerprint_seconds=dict(actual=cache_means[0],cached=cache_means[3],difference=cache_means[0]-cache_means[3]),
        checks=verification,bindings=bindings,threshold=1.05)
    write(BASE/'observations.json',value);write(FOLDER/'observations-20260920.json',value)
    tables={}
    for protocol in ['isolated','resident']:
        rows=[]
        for row in gaps:
            if row['protocol']==protocol:rows.append(f"| {row['case']} | {row['policy']} | {row['managed_ms']:.4f} | {row['ort_ms']:.4f} | {row['ratio']:.4f} | {row['deficit_ms']:.4f} | {row['required_reduction_percent']:.3f}% |")
        tables[protocol]='\n'.join(rows)
    profile_rows=[]
    for case in CASES:
        rows=[r for r in attribution if r['case']==case]
        profile_rows.append('| '+case+' | '+' | '.join('/'.join(f'{r["families_ms"][name]:.4f}' for r in rows) for name in ['MatMul','BiasGelu','MaskedSoftmax','LayerNormalization'])+' | '+'/'.join(f'{r["outside_nodes_ms"]:.4f}' for r in rows)+' |')
    norm_rows='\n'.join(f"| {r['case']} | {r['Product']:.6f} | {r['Wide']:.6f} | {r['reduction_ms']:.6f} | {r['reduction_percent']:.3f}% |" for r in normalization)
    header='| Case | Policy | Managed ms | ORT ms | Ratio | Reduction to 1.05, ms | Reduction of managed time |\n|---|---|---:|---:|---:|---:|---:|'
    report=f'''# Remaining e5 gap and tested mechanisms — September 20, 2026

**The retained evidence does not establish the 1.05 target or a hardware limit.**
Three of four primary lengths remain above the target in each descriptive
protocol below. Repeated low-level kernel variants have reached a plateau in
the tested forms. Two qualified optional mechanisms remain promising, but their
isolated full-model timing is unqualified and their defaults remain off.

This review recomputes the required reduction as
`max(0, managed_ms - 1.05 * ort_ms)` from unrounded retained data. It is the
remaining absolute budget under that observation, not a confidence bound or a
prediction. Eight tokens has zero remaining budget in these tables. Length512
is the regression/stretch case, not one of the four primary lengths.

The [isolated public-options comparison](../public-ort-20260919.md), source
`c6bf781`, has six complete processes per configuration and matching native
cohorts. All5,940 retained Execute/request samples are rechecked here against
their stored means; the table uses Execute only.

{header}
{tables['isolated']}

The [later resident-process A/A](../interleaved-processes/aa-results-20260920.md)
uses qualified `4f10e8b` and averages all three identical managed roles. Each
policy has its own matching native cohort. Its complete timing screen fails.
Resident and isolated processes are different measurement conditions: do not
average these tables, subtract one from the other, or infer a revision speedup.

{header}
{tables['resident']}

Both experiments use AMD EPYC9V74, CPU2, .NET10.0.8 and ORT1.23.2, with complete
numerical/ownership/resource checks. Neither supplies calibrated confidence.
The gap is small enough that order/runtime effects observed in the failed controls
matter. The historical negative results remain evidence; another unchanged
calibration run cannot establish a new interpretation.

The earlier managed profile (`8e93aa7`) retains all150profiled calls, two visits
per length. It explicitly enables the nine mechanisms later promoted and the
legacy `RELEASE_RESHAPE_VIEWS` override. These are historical managed node totals,
not a new public-Default profile, a native cost difference or a hardware bound.
All15samples in every reported family/visit are included and rechecked.

| Case | MatMul ms, visits1/2 | BiasGelu ms | MaskedSoftmax ms | LayerNorm ms | Outside nodes ms |
|---|---:|---:|---:|---:|---:|
{chr(10).join(profile_rows)}

MatMul remains the largest observed component. This explains the emphasis on
complete projection banks, but a two-percent bank improvement is only a
nomination threshold. It does not automatically cover the full-model deficit,
and a bank omits graph execution and real activation dependencies. Conversely,
the profile is too indirect to prove that bookkeeping or normalization cannot
contribute. Do not subtract its family totals from native unprofiled latency.

The [minimum-work LayerNorm comparison](../layernorm-minimum/results-20260920.md)
passes its controls and exact-output checks. These are complete captured
normalization banks, excluding the rest of the model:

| Bank | Product ms | Wider transform ms | Component reduction ms | Component reduction |
|---|---:|---:|---:|---:|
{norm_rows}

The [fingerprint component](../fingerprint-cache/results-20260920.md) drops from
{1000*cache_means[0]:.6f}ms to {1000*cache_means[3]:.6f}ms, a
{1000*(cache_means[0]-cache_means[3]):.6f}ms component difference. The separate
[common-state model comparison](../fingerprint-balanced/comparison-results-20260920.md)
passes its screen, while isolated deployment controls and the later resident
controls fail. These results have different state and execution boundaries.
Adding their component differences to a historical model mean would create an
unmeasured synthetic result. Neither switch is promoted by this review.

The source and complete experiments support the following decisions:

| Tested mechanism | Evidence and current decision |
|---|---|
| Wider row/column layouts | The48-column existing-layout variant regresses; contiguous packing confounds a useful narrow remainder. Fourteen rows increase register pressure and do not establish a complete primary-case benefit. These are covered in the [source review](../../../docs/source-review.md); choosing another tile size alone is not a new causal result. |
| Destination overwrite | [Complete clear/compute comparison](../packed-overwrite-20260919.md) preserves arithmetic but misses its fixed gain screen. It does not justify removing accumulation/clearing contracts. |
| Activation packing | [Index-based packing](../projection-input-pack/results-20260919.md) and the subsequent [literal-pointer version](../projection-input-pointer/results-20260919.md) both regress complete banks. The pointer version proves removal of the targeted address instructions; that fact alone does not establish speed. |
| Reduction blocking | [Fixed128/256 blocking](../projection-reduction-timing/results-20260920.md) preserves FMA order and passes stable controls, but both versions regress every complete bank. Do not import the branch's blocked-packing stack or repeat these versions unchanged. |
| Conditional GELU | [Actual AMD comparison](../gelu-uniform-amd/results-20260920.md) passes arithmetic/code proof but fails the fixed primary gains and worker regression screen. It stays out of production. |
| Fingerprint cache and wider LayerNorm | Both have qualified arithmetic and useful component evidence. Their pending issue is justified complete-model measurement, not another unchanged correctness replay. |

The inspected Microsoft source remains pinned at
`a83fc4d58cb48eb68890dd689f94f28288cf2278`. Its AVX512 microkernel uses24
accumulators for twelve rows across32columns; its SGEMM implementation separates
packed constant weights, panels and first-block overwrite. Lokad's current
twelve-row source shares that accumulator shape, while its compiler/addressing,
packing and graph contracts differ. These source differences locate hypotheses;
they do not prove which native dispatch runs or impose a lower latency bound.

The next e5 change must either identify a distinct remaining mechanism with a
cost large enough to matter and actual instruction evidence, or resolve the
interpretation of complete-model timing before promoting the already-qualified
optional mechanisms. The unchanged1.05target remains open. This review closes
only the evidence synthesis and establishes the scope of the tested plateau;
it does not close all optimization opportunities or the broader PLAN.md goal.

[Complete recomputed observations](observations-20260920.json) retain both full
tables, every profile family, component means and selected source/evidence hashes.
The script verifies selected summaries against their historical receipts; it does
not claim to rerun each original experiment or revalidate every historical file.
Run `C:/Python313/python.exe -X utf8 -B tests/e5/remaining-gap/review.py` only with
a fresh output directory; the original result is retained under
`artifacts/e5-remaining-gap-review-v2-20260920`. The first formatter assumed a
`Softmax` family instead of the actual `MaskedSoftmax`; its saved calculations,
source and failure record remain in the original directory. No new inference, VM workload,
product code, numerical tolerance or production default changes in this review.
'''
    with (FOLDER/'results-20260920.md').open('x',encoding='utf-8') as f:f.write(report)
    write(BASE/'receipt.json',dict(passed=True,bindings=bindings,outputs={p.relative_to(ROOT).as_posix():pin(p) for p in [BASE/'observations.json',FOLDER/'observations-20260920.json',FOLDER/'results-20260920.md',Path(__file__)]}))
    print(json.dumps(dict(passed=True,rows=len(gaps),profile_visits=len(attribution),bindings=len(bindings),receipt=pin(BASE/'receipt.json'))))


if __name__=='__main__':main()
