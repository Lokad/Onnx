"""Retain every tier and check the actual optimized register schedule."""
from pathlib import Path
import json,re,shutil,sys
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/pyannote-winograd-register-transform-codegen-amd-20260923'
DEST=ROOT/'artifacts/pyannote-winograd-register-transform-codegen-review-20260923'
sys.path.insert(0,str(ROOT/'tests/pyannote/winograd-register-transform-codegen'))
from protocol import pin,read,save
from listings import listings

def operations(body):
    return [line.strip() for line in body.splitlines() if re.match(r'^       [a-z]',line) and not line.strip().startswith('align')]

def main():
    assert not DEST.exists();assert pin(BASE/'closed.json')['sha256']=='e503b789c4a56d6aa8ea6659221e24c79d834f2ea042de9fcb559a7144f981f1'
    proof=read(BASE/'closed.json');assert proof['passed'] and proof['manual_review_pending']
    for name,wanted in proof['files'].items():assert pin(BASE/name)==wanted,name
    retained={};review=[];raw=[];optimized={}
    for role in ['current','candidate']:
        for width in [256,512]:
            name=f'{role}-captured-{width}';path=BASE/'collected'/name/'jit.asm';bodies=listings(path)
            assert len(bodies)==(1 if role=='current' else 3)
            assert all(b['complete_uninterleaved'] and not b['managed_stdout_repairs'] for b in bodies)
            retained[name]=bodies;target=OUT/(name+'.asm')
            if target.exists():assert pin(target)==pin(path), 'Existing raw listing must remain byte-exact'
            else:shutil.copyfile(path,target)
            raw.append(dict(file=target.name,pin=pin(target)))
            selected=[b for b in bodies if b['tier']=='Tier0-FullOpts' or b['tier'].startswith('Tier1')]
            assert len(selected)==1
            b=selected[0];body=b['body'];instructions=operations(body);optimized[name]=instructions
            if role=='current':
                assert b['code_bytes']==[598] and 'sub      rsp, 512' in body
                assert len(re.findall(r'\bvpermps\b',body))==4 and len(re.findall(r'\bvperm2f128\b',body))==4
                workspace='rax' if width==256 else 'rbx'
                stores=re.findall(r'vmovups\s+ymmword ptr \['+workspace+r'\+r\d+\], ymm\d+',body)
                loads=re.findall(r'vmovups\s+ymm\d+, ymmword ptr \['+workspace+r'\+r\d+\]',body)
                assert len(stores)==len(loads)==4
                review.append(dict(name=name,tier=b['tier'],bytes=598,workspace_bytes=512,workspace_register=workspace,workspace_vectors_per_channel=16))
                continue
            assert b['tier']=='Tier1' and b['code_bytes']==[842]
            start=body.index('G_M000_IG03:');end=body.index('G_M000_IG06:');loop=body[start:end]
            counts={op:len(re.findall(r'\b'+op+r'\b',loop)) for op in ['vpermps','vperm2f128','vaddps','vsubps','vmovups']}
            assert counts==dict(vpermps=16,vperm2f128=16,vaddps=8,vsubps=24,vmovups=16)
            assert not re.search(r'\[(?:rbp|rsp)(?:[+\-\]])',loop)
            assert not re.search(r'\bvfmadd|\bcall\b|\b(?:i?div)\b',body)
            assert 'sub      rsp, 512' not in body
            loads=re.findall(r'vpermps\s+ymm\d+, ymm0, ymmword ptr \[([^\]]+)\]',loop)
            assert len(loads)==16
            assert [x.split('+')[0] for x in loads]==['rbx']*8+['r14']*4+['rsi']*4
            assert [x.split('+',1)[1] if '+' in x else '0' for x in loads]==['0','0x20','0x08','0x28']*4
            assert re.findall(r'vperm2f128[^\n]+, (\d+)',loop)==['32','49','32','49']*4
            assert 'lea      ebx, [r10+0x02]' in loop and 'lea      r14d, [r10+0x01]' in loop and 'lea      r14d, [r10+0x03]' in loop
            stores=[line.strip() for line in loop.splitlines() if re.search(r'vmovups\s+ymmword ptr',line)]
            assert len(stores)==16 and all('[rcx+4*' in line for line in stores)
            registers=sorted({int(n) for n in re.findall(r'\bymm(\d+)\b',loop)})
            assert registers==list(range(10))
            assert loop.count('jl       G_M000_IG03')==1 and 'inc      edx' in loop and 'cmp      edx, r8d' in loop
            review.append(dict(name=name,tier=b['tier'],bytes=842,workspace_bytes=0,counts=counts,registers=registers,
                vector_spills=[],arithmetic_calls=[],input_row_order=[0,2,1,3],input_offsets_floats=[0,8,2,10],output_vectors=16,
                manually_reviewed='All sixteen destination planes retain ((plane*c+ic)*8) offsets; each vertical expression preserves row0-row2, row1+row2, row2-row1, row1-row3 in the original column order.'))
    assert optimized['candidate-captured-256']==optimized['candidate-captured-512']
    DEST.mkdir();value=dict(passed=True,mechanism_admitted=True,no_performance_measurement=True,timing_deferred='User priority changed to Parakeet; candidate remains unselected.',closure=pin(BASE/'closed.json'),reviewer=pin(Path(__file__)),raw_files=raw,review=review,all_bodies=retained,
        costs=dict(current_optimized_bytes=598,candidate_optimized_bytes=842,candidate_instrumented_tier0_bytes=2637,
            cold_tier_note='Removing localloc changes immediate Tier0-FullOpts to instrumented Tier0 followed by Tier1. Both duplicate initial bodies are retained per mode; complete-call warmup clocks and ordinary tiering must remain in the screen.'))
    save(DEST/'review.json',value);save(OUT/'codegen-observations-20260923.json',value)
    (OUT/'codegen-20260923.md').write_text('''# Register-scheduled input transform: generated-code review

**Generated code reviewed; timing deferred while Parakeet takes priority.** The candidate remains unselected. The actual candidate
DLL eliminates the contiguous transform's 512-byte stack workspace. Its optimized
channel loop uses ten YMM registers, including the permutation mask, with no stack
references, arithmetic calls or vector spills. AVX512-disabled and ordinary
execution emit identical optimized instruction sequences for this method.

The loop retains sixteen original input loads folded into `vpermps`, sixteen
`vperm2f128` operations, eight additions, twenty-four subtractions and sixteen
final stores. Rows are loaded in order 0,2,1,3. Each row uses the original float
offsets 0,8,2,10 and lane controls 32,49,32,49. Destination planes preserve
`(plane*c+ic)*8`; vertical expressions remain row0-row2, row1+row2, row2-row1,
row1-row3 for each original column. The channel counter still advances by one.
The captured full numerical rows independently preserve the current product bits.

There are costs to measure. Optimized code grows from 598 to 842 bytes. Removing
`stackalloc` also changes the JIT path: the current method starts in optimized
Tier0-FullOpts, while the candidate emits 2,637-byte instrumented Tier0 bodies
before its optimized Tier1 body. All eight emitted bodies are retained, including
both duplicate initial bodies per candidate mode. Cold instrumented code contains
stack traffic; this is not a cold-start speed claim. The screen must keep ordinary
tiering and all fixed warmup clocks, without calibration or targeted pre-jitting.

All four capture workers and 348 captured calls pass their complete numerical
checks. All 167 resource observations pass; peak owned RSS is 575,193,088 bytes.
Owner809798 / birth1790152841.04 and every worker are terminal. A single exact
method filter keeps all listings complete without interleaving or repairs.

[Every tier and review check](codegen-observations-20260923.json),
[current AVX2](current-captured-256.asm), [current ordinary](current-captured-512.asm),
[candidate AVX2](candidate-captured-256.asm), [candidate ordinary](candidate-captured-512.asm).

Capture closure: `e503b789c4a56d6aa8ea6659221e24c79d834f2ea042de9fcb559a7144f981f1`.
No timing result, application speedup or root integration follows from this review.
''',encoding='utf8')
    print(json.dumps(dict(review=pin(DEST/'review.json'),bodies=sum(len(v) for v in retained.values()),optimized=len(review))))

if __name__=='__main__':main()
