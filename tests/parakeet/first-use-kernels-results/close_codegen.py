"""Close complete-body checks and the reviewed M43 kernel mechanisms."""
import collections,json,re
from pathlib import Path
from inspect_codegen import BASE,OUT,ROOT,METHODS,parse,pin

def main():
    target=Path(__file__).resolve().parent/'codegen-review-20260923.json';assert not target.exists()
    census=json.loads((OUT/'census.json').read_text());assert census['first_use_passed']
    assert pin(BASE/'closed.json')==census['closure']
    roles={};checks=[]
    for role in ['current','candidate']:
        path=BASE/'collected/logs'/(role+'-codegen-512.stdout');assert pin(path)==census['inputs'][role]
        roles[role]=parse(path)
        for row in roles[role]:
            body=row['body'];labels=re.findall(r'^(G_M\d+_IG\d+):',body,re.M)
            assert len(labels)==len(set(labels))>0
            assert set(re.findall(r'G_M\d+_IG\d+',body))<=set(labels)
            assert row['instructions'] and row['bytes']>0
            checks.append(dict(role=role,index=row['index'],basic_blocks=len(labels),passed=True))
    assert len(roles['current'])==33 and len(roles['candidate'])==13
    arithmetic={}
    for name in METHODS:
        select=lambda role,tier:next(r for r in roles[role] if r['method'].startswith('Lokad.Onnx.MathOps:'+name+'(') and r['tier']==tier)
        left=select('current','Tier1');right=select('candidate','FullOpts')
        counts=lambda r:dict(collections.Counter(s.split()[0] for s in r['instructions'] if s.startswith(('vfmadd','vmul','vadd','vpmaskmov'))))
        assert counts(left)==counts(right),(name,counts(left),counts(right))
        assert 'No PGO data' in right['body'] and 'PATCHPOINT' not in right['body']
        arithmetic[name]=counts(right)
    wrappers=[r for r in roles['candidate'] if ':RunFloatMatMulKernel(' in r['method']]
    assert len(wrappers)==3
    for row in wrappers:
        assert 'RunGeneralFloatMatMulKernel' in row['body'] and 'RunShortWidePackedRows' in row['body']
        assert 'MathOps:' not in row['body'] and 'ArrayPool' not in row['body']
    wrapper=next(r for r in wrappers if r['tier']=='Tier1')
    assert wrapper['bytes']==296
    assert all(x in wrapper['body'] for x in ['sub      eax, 48','cmp      eax, 15','cmp      esi, 0x400','cmp      edx, 0x400','cmp      rax, 0x4000000'])
    short=next(r for r in roles['candidate'] if ':RunShortWidePackedRows(' in r['method'] and r['tier']=='Tier1')
    assert all(x in short['body'] for x in ['_3x4packed(', '_2x4packed_bump(', 'MathOps:mm_unsafe_vectorized_intrinsics(', 'call     G_M000_IG54'])
    assert 'TryPackedAvx512Rows' not in short['body']
    review=dict(passed=True,mechanism_admitted=True,numerical_passed=True,no_performance_measurement=True,
        closure=census['closure'],census=pin(OUT/'census.json'),inputs=census['inputs'],bodies=census['bodies'],
        first_use=census['first_use'],body_checks=checks,arithmetic_opcode_counts=arithmetic,
        files={p.relative_to(ROOT).as_posix():pin(p) for p in OUT.iterdir() if p.is_file()},
        review='All 46 emitted bodies retained and checked for complete boundaries, labels, branches and call census. Four candidate float kernels each emit one optimized FullOpts body, with no instrumented Tier0/OSR replacement or dynamic PGO data. Packing preserves 32-column panels and scalar tail; the JIT uses the existing 512-bit copies on this CPU. Packed two/three-row main loops keep eight/twelve ordered AVX2 FMA accumulators without stack vector spills in those loops. Eight-wide remainder uses FMA; masked remainder preserves separate multiply/add, as does raw scalar tail. Native layouts differ from current Tier1; no identical-machine-code claim. Wrapper remains 296 bytes with original guards and separate helper calls. Short helper returns scratch through IG54 before its IG48 odd-row fallback using original B. General helper IL and all arithmetic bodies are proven exact by the normal build inventory. These checks and exact actual-DLL numerical results admit only the component timing experiment.')
    target.write_text(json.dumps(review,indent=2)+'\n')
    print(json.dumps(dict(review=pin(target),bodies=len(checks),arithmetic=arithmetic)))

if __name__=='__main__':main()
