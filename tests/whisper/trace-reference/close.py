"""Report independently verified selected-case boundary observations."""
import datetime,json,shutil
from pathlib import Path
from analyze import ROOT,BASE,TRACE,REF,pin,read,write


def main():
    observations=read(BASE/'observations.json');verification=read(BASE/'independent-verification.json')
    assert observations['structural_passed'] and verification['passed'] and verification['comparisons']==1312
    rows=[json.loads(line) for line in (BASE/'comparisons.jsonl').read_text().splitlines()]
    display=[]
    for path in observations['paths']:
        a=path['references']['numpy'];b=path['references']['ort'];assert a['first_failure']==b['first_failure'] and a['failed_boundaries']==b['failed_boundaries']
        layer=a['first_failure']-8;selected=[r for r in rows if (r['request'],r['kind'],r['reference'])==(path['request'],path['kind'],'numpy')]
        display.append(f"| {path['request']}: {path['name']} | {path['kind']} | {layer} | {selected[27]['max_scaled']:.8g} | {selected[28]['max_scaled']:.8g} | {a['final']['max_scaled']:.8g} | {a['final']['failed_values']:,} |")
    assert [p['references']['numpy']['first_failure'] for p in observations['paths']].count(28)==15
    assert [p['references']['numpy']['first_failure'] for p in observations['paths']].count(30)==1
    assert all(r['failed_values']==0 for r in rows if r['index']<=27)
    final=[r for r in rows if r['index']==40]
    assert len(final)==32 and all(r['failed_values'] for r in final)
    disagreements=max(abs(a['max_scaled']-b['max_scaled']) for a,b in zip(rows[::2],rows[1::2],strict=True))
    folder=Path(__file__).resolve().parent;report=folder/'results-20260920.md';data=folder/'observations-20260920.json'
    write(data,dict(observations,independent_verification=verification,reference_maximum_error_difference=disagreements,comparison_count=observations['comparisons'],comparisons=rows))
    text=f'''# Selected Whisper traces against both float64 references

Both FP32 engines exceed the unchanged scaled-error limit `1e-4` against both
independent float64 references. All saved boundaries through encoder layer 19
pass in these selected cases. Fifteen of sixteen engine/input paths first fail
at the saved output of layer 20; managed execution on managed features for
`908-157963-0000` first fails at layer 22. Both reference implementations identify
the same first failing boundary and the same number of failing boundaries.

This is a local analysis of previously retained arrays, with **no new model
inference**. It covers all 41 saved boundaries, both inference engines, both
feature producers, three previously selected natural cases and a repeat of the
first: 656 FP32 arrays compared against both references, for 1,312 comparisons.
Every value is included, including padded frames. The selection remains the
original first case and previously nominated largest-difference cases; it is
not an independent accuracy corpus or a complete-corpus intermediate result.

The table uses the NumPy/SciPy float64 reference for displayed magnitudes. The
complete JSON retains both it and the separate ORT-float64/Python-erf reference.
Maximum reported scaled-error magnitudes differ between these references by at
most `{disagreements:.9g}`. The original full-reference comparison had already
verified agreement within `1e-9` across every retained reference output.

| Request | Engine/features | First failing layer | Layer 19 maximum | Layer 20 maximum | Final maximum | Final failed values |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(display)}

In MM/MN/NM/NN, the first letter names the inference engine and the second names
the feature producer; M is managed and N is native. Layer numbers are the ONNX
graph's zero-based indices. Every final array has 1,920,000 values. Scaled error
is `abs(actual-reference)/max(1,abs(reference))`; a value fails strictly above
`1e-4`. All intermediate first-case repeats reproduce identical FP32 bits.

These are **instrumented trace outputs**. Exposing 41 outputs preserves the
managed final output's exact bits, but changes native final outputs slightly
relative to the original uninstrumented encoder. That previously measured
instrumentation difference is below `1e-4` and remains recorded for every path.
The current native maxima must therefore not replace the original full-corpus
uninstrumented FP32/reference results. Both engines' final failures remain.

The first saved failing boundary does not identify an incorrect operator.
Existing [layer-20 crossed-input evidence](../layer20-cross/results-20260920.md)
shows both engines can amplify incoming differences while all same-input local
intermediate comparisons pass. The present independent references also reject
using either FP32 engine as numerical truth. A causal investigation must distinguish
incoming accumulated error from local arithmetic error; no tolerance, production
default or claim of transcript equivalence is changed here.

## Verification and retained evidence

All 1,340 used files are checked against their original closed receipts before
analysis and again independently: {observations['input_bytes']:,} bytes of selected
arrays, manifests, results and receipts. Matching uses original request index,
name, feature hash, output name, shape and order. Both original campaigns remain
closed and unchanged. The original full corpus and its failures are documented
in the [complete reference report](../full-reference/results-20260920.md).

The chunked comparison takes {observations['seconds']:.2f} seconds with
{observations['maximum_rss']:,} bytes maximum observed RSS over 656 resource checks.
A separate full-array implementation independently recomputes every comparison,
including maxima, coordinates, counts and RMS, and checks every path summary:
{verification['scalar_checks']:,} scalar checks, {verification['seconds']:.2f} seconds,
{verification['peak_rss']:,} bytes peak observed RSS. Both obey the prospective
900-second, 2 GiB RSS and 1 GiB available-memory limits.

Artifact: `artifacts/whisper-trace-reference-20260920`.
Full observations: [JSON](observations-20260920.json).
The artifact binds the original input receipts and every selected array hash,
all 1,312 comparison records, the independent verification and exact analysis
source snapshots. No VM workload, model download or inference replay occurred.
'''
    with report.open('x',encoding='utf-8') as f:f.write(text)
    snapshots=BASE/'source';snapshots.mkdir()
    for p in folder.glob('*.py'):shutil.copyfile(p,snapshots/p.name)
    shutil.copyfile(ROOT/'.agent/m4-whisper-trace-reference-20260920.md',snapshots/'prospective-plan.md')
    files={p.relative_to(ROOT).as_posix():pin(p) for p in sorted(BASE.rglob('*')) if p.is_file()}
    files.update({p.relative_to(ROOT).as_posix():pin(p) for p in [report,data]})
    write(BASE/'closed.json',dict(analysis_passed=True,numerical_qualification=False,closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,input_index=pin(BASE/'inputs.json')))
    print(json.dumps(dict(receipt=pin(BASE/'closed.json'),files=len(files),comparisons=len(rows))))


if __name__=='__main__':main()
