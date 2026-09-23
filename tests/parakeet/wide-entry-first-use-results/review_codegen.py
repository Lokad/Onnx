"""Pin complete capture, identical arithmetic, and reviewed wide entry/helper paths."""
import json
import re
from pathlib import Path
from inspect_codegen import ROOT, BASE, OUT, METHODS, parse, pin


def normalized(row):
    labels = dict(re.findall(r'^(G_M\d+_IG\d+):\s+;; offset=(0x[0-9A-Fa-f]+)', row['body'], re.M))
    result = []
    for instruction in row['instructions']:
        instruction = re.sub(r'0x[0-9A-Fa-f]{10,}', '<address>', instruction)
        instruction = re.sub(r'G_M\d+_IG\d+', lambda m: labels[m[0]], instruction)
        result.append(re.sub(r'for IG\d+', 'padding', instruction))
    return result


def main():
    target = Path(__file__).parent / 'codegen-review-20260923.json'; assert not target.exists()
    prior = ROOT / 'tests/parakeet/wide-projection-isolation-results/codegen-review-20260923.json'
    assert pin(prior)['sha256'] == '3176d6cc3312083ae2ccdc34e23d9647016f960a2276d94e30c226699b364f67'
    previous = json.loads(prior.read_text()); assert previous['passed'] and previous['mechanism_admitted']
    for n,v in previous['files'].items(): assert pin(ROOT/n)==v,n
    census=json.loads((OUT/'census.json').read_text());structure=json.loads((OUT/'structure.json').read_text())
    assert census['first_use_passed'] and structure['passed'] and structure['census']==pin(OUT/'census.json')
    actual=parse(BASE/'collected/logs/candidate-codegen-512.stdout')
    oldpath=ROOT/'artifacts/parakeet-wide-projection-isolation-numerics-amd-v2-20260923/collected/logs/candidate-codegen-512.stdout'
    old=parse(oldpath); comparisons=[]
    for name in METHODS.values():
        left=[r for r in old if r['method'].startswith('Lokad.Onnx.MathOps:'+name+'(')]
        right=[r for r in actual if r['method'].startswith('Lokad.Onnx.MathOps:'+name+'(')]
        assert len(left)==len(right)==1 and left[0]['tier']==right[0]['tier']=='FullOpts'
        assert normalized(left[0])==normalized(right[0])
        comparisons.append(dict(method=name,old_body=left[0]['index'],new_body=right[0]['index'],bytes=right[0]['bytes'],normalized_instruction_text_equal=True))
    entry=next(r for r in actual if ':RunWideProjectionMatMul2DCore(' in r['method'])
    helper=next(r for r in actual if ':RunIsolatedShortWidePackedRows(' in r['method'])
    dispatch=next(r for r in actual if ':DispatchWideProjectionMatMul2DCore(' in r['method'] and r['tier']=='Tier1')
    assert entry['tier']==helper['tier']=='FullOpts' and entry['bytes']==5625 and helper['bytes']==567 and dispatch['bytes']==563
    assert all(n in entry['body'] for n in ['TensorExecutionOptions:Validate','HasStandardStrides','SharesBackingMemory','ClearWithoutReferences','ResolvePackedKernel','DensifyFloatOperands','ParallelOptions:set_MaxDegreeOfParallelism','RunIsolatedShortWidePackedRows','RunFloatMatMulKernel'])
    assert all(n in helper['body'] for n in METHODS.values()) and 'TryPackedAvx512Rows' in helper['body']
    normal=helper['body'].split('G_M000_IG18:')[1].split('G_M000_IG20:')[0]
    assert normal.index('SharedArrayPool`1[float]:Return') < normal.index('G_M000_IG19:') < normal.index('mov      r8, qword ptr [rbp-0x30]') < normal.index('ShortWideMultiplyRemainder')
    assert 'SharedArrayPool`1[float]:Return' in helper['body'].split('G_M000_IG29:')[1]
    assert sum('tail.jmp' in s for s in dispatch['instructions'])==2
    assert 'cmp      rax, 0x4000000' in dispatch['body'] and 'cmp      dword ptr [r10], 48' in dispatch['body']
    assert dispatch['body'].count(', 0x400\n')==2
    notes=dict(entry='5625-byte first-use FullOpts exact IL clone. Reviewed validation before clearing, prepared-weight resolution, densification, pin/disposal paths, parallel setup and isolated-guard/fallback calls. New parallel callback proven by exact IL and complete Parallel(2) numerics.',
        helper='567-byte first-use FullOpts keeps packer and arithmetic as calls. IG18 returns scratch before IG19 reads original B at rbp-0x30 and calls odd-row remainder. Exceptional finally IG29 also returns scratch. Optional AVX512 branch and two/three-row grouping retained.',
        arithmetic='All four standalone FullOpts copied-kernel instruction streams equal the reviewed M52 copies after address masking/local branch-offset resolution. Ordered AVX2 accumulation and separate multiply/add tails remain.',
        dispatcher='563-byte Tier1 dispatcher retains null/rank/axis/budget/options guards and two tail calls; first-use Tier0 and instrumented versions are also retained. FMA hardware check folds true on this AMD CPU.',
        scope='All 51 emitted bodies retained with complete label checks. Manual focus is new entry/helper/dispatcher; arithmetic verified against prior complete review. This independent capture does not identify code executed in a scored process or prove a performance improvement.')
    files={p.relative_to(ROOT).as_posix():pin(p) for p in OUT.rglob('*') if p.is_file()}
    for p in [Path(__file__),Path(__file__).parent/'inspect_codegen.py',Path(__file__).parent/'review_structure.py',Path(__file__).parent/'entry-il-review-20260923.json',prior,oldpath]:files[p.relative_to(ROOT).as_posix()]=pin(p)
    value=dict(passed=True,mechanism_admitted=True,no_performance_measurement=True,closure=pin(BASE/'closed.json'),
        numerical_values_per_worker=7749681,numerical_groups_per_worker=68,complete_bodies=len(census['bodies']),
        comparisons=comparisons,entry=dict(index=entry['index'],bytes=entry['bytes']),helper=dict(index=helper['index'],bytes=helper['bytes']),dispatcher=dict(index=dispatch['index'],bytes=dispatch['bytes']),notes=notes,files=files)
    target.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(review=pin(target),complete_bodies=value['complete_bodies'],mechanism_admitted=True,comparisons=comparisons)))


if __name__=='__main__':main()
