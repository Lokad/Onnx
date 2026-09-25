"""Record concrete native differences and the bounded dispatch-isolation hypothesis."""
import collections
import difflib
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tests/benchmarks/e5-direct-code-diagnostic-amd'))
from protocol import pin,read
BASE=ROOT/'artifacts/e5-direct-code-diagnostic-amd-20260925'


def main():
    assert not (OUT/'inspection-20260925.json').exists()
    assert pin(BASE/'closed.json')['sha256']=='de164f33d9d75d089905a54d95e6b08971d6644745d4959017f689079682bd92'
    assert pin(OUT/'observations-20260925.json')['sha256']=='188c2d9c6d8ead177d8dd7a2350d207906588a44784fb7f265944518455a1b8c'
    published=read(OUT/'observations-20260925.json');rows=[];inputs={}
    for role,report in published['reports'].items():
        for row in report['code_versions']:
            path=OUT/row['listing_path'];assert pin(path)==row['listing_file'];inputs[path.relative_to(ROOT).as_posix()]=pin(path)
            listing=path.read_text();stack=re.findall(r'^\s+[0-9A-F]+\s+sub\s+rsp, (\d+)\s*$',listing,re.M)
            if row['tier']=='Tier1':assert stack
            owned_calls=[c for c in row['calls'] if any(n in c for n in ['TryRunOwnedPackedBatches','OwnedPackedTensor:Resolve','RunOwnedPackedRows'])]
            rows.append(dict(process=role,product=report['product'],method=row['method'],tier=row['tier'],runtime_tier=row['runtime_tier'],
                native_bytes=row['native_bytes'],printed_instruction_bytes=row['instruction_bytes'],
                unprinted_bytes=row['native_bytes']-row['instruction_bytes'],stack_subtractions=[int(x) for x in stack],
                comments=row['comments'],owned_route_calls=owned_calls,listing=row['listing_path'],calls=row['calls']))
    final={(r['process'],r['method']):r for r in rows if r['tier']=='Tier1'}
    for role in 'bc':
        row=final[(role,'RunBatchedFloatMatMul')]
        assert any('RunOwnedPackedRows' in c for c in row['owned_route_calls'])
        assert not any('TryRunOwnedPackedBatches' in c for c in row['calls'])
    for role in 'ad':assert not final[(role,'RunBatchedFloatMatMul')]['owned_route_calls']
    comparisons=[]
    for method in ['RunFloatMatMulKernel','RunBatchedFloatMatMul']:
        for first,second in [('a','d'),('b','c'),('a','b')]:
            def normalize(text):return re.sub(r'G_M\d+_IG\d+','LOCAL',text)
            left=collections.Counter(map(normalize,final[(first,method)]['calls']))
            right=collections.Counter(map(normalize,final[(second,method)]['calls']))
            comparisons.append(dict(method=method,before=first,after=second,added_calls=dict(right-left),removed_calls=dict(left-right)))
    source=ROOT/'artifacts/parakeet-direct-depthwise-source-v2-20260925/source/src/Lokad.Onnx/TensorOps.MatMul.cs'
    selected=ROOT/'src/Lokad.Onnx/TensorOps.MatMul.cs'
    current=source.read_text();release=selected.read_text()
    insertion='        if (TryRunOwnedPackedBatches(bx, by, z, options)) return;\n'
    assert current.count(insertion)==1 and current.replace(insertion,'')==release
    assert current.count('RunBatchedFloatMatMul(bx, by, target, options);')==1
    assert current.count('RunBatchedFloatMatMul(bx, by, z, options);')==1
    for p in [Path(__file__),OUT/'observations-20260925.json',BASE/'closed.json',source,selected]:inputs[p.relative_to(ROOT).as_posix()]=pin(p)
    result=dict(passed=True,diagnostic_only=True,new_inference=False,retrospective_cause_established=False,
        optimization_settings_changed=False,inputs=inputs,rows=rows,call_differences=comparisons,
        next_hypothesis='Move the existing owned-packed decision to the two immediate callers; restore the shared RunBatchedFloatMatMul release body exactly, without changing helper arithmetic, shapes, order or flags.')
    with (OUT/'inspection-20260925.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
    lines=['# Isolate the packed-weight decision from the shared batched dispatcher','',
        'The candidate inserts TryRunOwnedPackedBatches at the start of the shared',
        'RunBatchedFloatMatMul method. In both final candidate listings, the helper',
        'has been inlined: its shape/option tests and prepared-weight branch appear',
        'inside the dispatcher, including calls to OwnedPackedTensor.Resolve and',
        'RunOwnedPackedRows. Both release listings lack that route.','',
        '| Process | Product | Initial batched bytes | Intermediate batched bytes | Final batched bytes | Final entry stack subtraction |',
        '|---|---|---:|---:|---:|---:|']
    for role in 'abcd':
        group=[r for r in rows if r['process']==role and r['method']=='RunBatchedFloatMatMul'];assert len(group)==3
        lines.append(f"| {role} | {group[0]['product']} | {group[0]['native_bytes']} | {group[1]['native_bytes']} | {group[2]['native_bytes']} | {group[2]['stack_subtractions'][0]} |")
    lines+=['','The candidate adds exactly 143 bytes to each initial/intermediate batched',
        'version. The final candidate bodies are about 1.4 KB larger in this capture',
        'and reserve more stack. This is a concrete product difference. It is not',
        'proof that this difference caused the older 7.77% regression.','',
        'Fresh same-product processes also differ independently. Release A inlines',
        'GCHandle.Free internals where release D calls GCHandle.Free; candidate B/C',
        'retain the same call targets apart from register selection. All final target',
        'listings describe Synthesized PGO. The matrix dispatcher has the same',
        'source in both products; its native sizes still vary. Avoid interpreting',
        'all code-size variation as one product change.','',
        'One bounded intervention follows: remove the owned-packed decision from',
        'RunBatchedFloatMatMul and place the same decision immediately before each',
        'of its two existing calls. Keep all arguments, destination clearing, helper',
        'bodies, arithmetic, model preparation and implementation flags unchanged.',
        'This should restore the shared method\'s original compiled body while',
        'retaining the successful Parakeet route. Prove that structural prediction',
        'first, then run the original correctness and performance gates once on the',
        'new candidate. Reject it if those gates fail; do not search adjacent variants.','',
        'The listing reader needed one additive naming correction: the first batched',
        'body prints Instrumented Tier0 but its CLR load reports QuickJitted. Both',
        'labels, profile-counter calls and OSR patchpoints are preserved, and all',
        'same-process identity/order/size checks remain exact. Microsoft describes',
        'initial instrumentation for loop methods in its',
        '[instrumented-tier design](https://github.com/dotnet/runtime/blob/main/docs/design/features/DynamicPgo-InstrumentedTiers.md).','',
        'Byte-output limit: the four final batched listings omit 22–26 bytes relative',
        'to their printed group offsets and native-size totals. The printed bytes',
        'are therefore not a complete memory image. Preserve those gaps; do not',
        'invent padding or claim byte-for-byte reconstruction. Complete textual',
        'listings and exact CLR code-size associations remain available. No further',
        'capture is needed to establish the extra owned route and frame growth.','',
        '[All listings and clocks](report-20260925.md), [inspection evidence](inspection-20260925.json).','',
        'No score, release admission, runtime optimization setting or warmup policy changes.']
    with (OUT/'inspection-20260925.md').open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(inspection=pin(OUT/'inspection-20260925.json'),versions=len(rows),
        batched_final={r:dict(bytes=final[(r,'RunBatchedFloatMatMul')]['native_bytes'],stack=final[(r,'RunBatchedFloatMatMul')]['stack_subtractions'][0]) for r in 'abcd'})))


if __name__=='__main__':main()
