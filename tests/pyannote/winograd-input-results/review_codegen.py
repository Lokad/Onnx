"""Review all emitted optimized reductions; retain every raw tier."""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/pyannote-winograd-input-codegen-amd-20260923'
OUT=ROOT/'artifacts/pyannote-winograd-input-codegen-review-20260923'
REPORT=Path(__file__).resolve().parent

def pin(p):return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(p):return json.loads(p.read_text(encoding='utf8'))

def main():
    assert not OUT.exists()
    assert pin(BASE/'closed.json')['sha256']=='67a2d57445ad280193dc7e470e5caf132bda8bf234aa8d98cb9c9a1681c33cb3'
    for name,wanted in read(BASE/'closed.json')['files'].items():assert pin(BASE/name)==wanted,name
    code=read(BASE/'listings.json');reviews=[];all_bodies=[];inputs=[]
    for role,bodies in code.items():
        width=int(role.split('-')[1]);reg='ymm' if width==256 else 'zmm'
        reductions=[b for b in bodies if ':MultiplyWinograd'+str(width)+'(' in b['method'] and b['tier'].startswith('Tier1')]
        assert len(reductions)==3
        for row in reductions:
            block,=row['reductions'];body=block['body']
            assert block['fma_instructions']==8 and not block['vector_stack_references'] and not block['scalar_stack_references']
            assert not block['integer_multiplies'] and not block['arithmetic_shifts']
            assert re.findall(r'\bvfmadd231ps\s+('+reg+r'\d+),',body)==[reg+str(i) for i in range(8)]
            assert len(re.findall(r'\bvmovups\s+'+reg+r'8, '+reg+r'word ptr',body))==1
            assert not re.search(r'^\s+call\s',row['body'],re.M)
            assert len(re.findall(r'\{1to16\}',body))==(8 if width==512 else 0)
            assert block['input_broadcasts']==(8 if width==256 else 0)
            assert re.search(r'\badd\s+r(?:8|11), 32\b',body)
            reviews.append(dict(width=width,tier=row['tier'],bytes=row['code_bytes'],fmas=8,
                accumulators=[reg+str(i) for i in range(8)],weight_loads=1,vector_loop_spills=0,
                scalar_loop_frame_references=0,inline_memory_broadcasts=8 if width==512 else 0,
                explicit_broadcasts=block['input_broadcasts'],calls=0,body_sha256=row['body_sha256']))
        output,=[b for b in bodies if ':OutputWinograd'+str(width)+'(' in b['method'] and b['tier']=='Tier1']
        assert len(re.findall(r'\bvaddps\b',output['body']))==12
        assert len(re.findall(r'\bvsubps\b',output['body']))==12
        assert not re.search(r'\bvfmadd|^\s+call\s',output['body'],re.M)
        for name in ['PrepareWinograd','TransformWinogradInput']:
            found=[b for b in bodies if ':'+name+'(' in b['method']]
            assert found and all('vfmadd' not in b['body'] for b in found)
        transform,=[b for b in bodies if ':TransformWinogradInput(' in b['method']]
        assert transform['tier']=='Tier0-FullOpts'
        body=transform['body'];channel=body[body.index('G_M000_IG17:'):body.index('G_M000_IG22:')]
        assert not re.search(r'\bidiv\b|\bvfmadd|^\s+call\s',channel,re.M)
        assert len(re.findall(r'\bidiv\b',body))==2
        assert len(re.findall(r'\bvgatherdps\b',channel))==4
        assert len(re.findall(r'\bvmovaps\s+ymm[3-6], ymm[2-5]',channel))==4
        assert len(re.findall(r'\bvaddps\b',channel))==2
        assert len(re.findall(r'\bvsubps\b',channel))==6
        assert re.search(r'jne\s+G_M000_IG18',channel)
        assert re.search(r'jl\s+G_M000_IG20',channel) and re.search(r'jl\s+G_M000_IG17',channel)
        assert len(re.findall(r'\bsub\s+rsp, 512',body))==3
        assert body.count('CORINFO_HELP_MEMZERO')==(3 if width==256 else 0)
        assert len(re.findall(r'\bvmovdqu32\s+zmmword ptr',body))==(24 if width==512 else 0)
        inputs.append(dict(width=width,tier=transform['tier'],bytes=transform['code_bytes'],
            body_sha256=transform['body_sha256'],coordinate_divisions_per_batch=16,
            coordinate_divisions_in_channel_loop=0,masked_gathers_per_channel=16,
            vector_adds_per_channel=8,vector_subtracts_per_channel=24,
            explicit_stack_storage=1536,stack_initialization_helpers=3 if width==256 else 0,
            inline_stack_zero_vector_stores=24 if width==512 else 0,
            gather_mask_copies_per_channel=16,channel_loop_calls=0,
            channel_scalar_frame_references=[line.strip() for line in channel.splitlines() if re.search(r'\[(?:rbp|rsp)[+\-]',line)]))
        for index,row in enumerate(bodies):
            assert row['complete_body'] and row['complete_uninterleaved'] and row['body']==row['raw_body']
            name=f'{role}-{index:02}.asm';assert not (REPORT/name).exists()
            (REPORT/name).write_text(row['raw_body'],encoding='utf8',newline='\n')
            all_bodies.append(dict(role=role,file=name,method=row['method'],tier=row['tier'],bytes=row['code_bytes'],digest=pin(REPORT/name)))
    value=dict(passed=True,closure=pin(BASE/'closed.json'),reviewer=pin(Path(__file__)),reductions=reviews,
        input_transforms=inputs,all_bodies=all_bodies,all_tiers_retained=True,no_performance_measurement=True,
        manual_review='All six optimized reductions retain eight independent accumulators, one shared weight load and eight ordered channel FMAs, without vector spills, scalar frame references or calls inside reduction. AVX512 uses eight memory broadcasts; AVX256 uses eight explicit broadcasts. OSR entry imports remain visible. Full Tier1 inverse transforms have 12 vector adds and 12 subtracts with guarded odd stores and two divisions per tile/channel block. Both new input bodies hoist all coordinate divisions before the channel loop, use four masked gathers in a four-row loop, copy destructive gather masks, and perform ordered eight-lane arithmetic with explicit stack row storage. All transformed lanes are written. Three 512-byte stack allocations are still zero-initialized per batch: three MEMZERO helpers with AVX512 disabled or 24 inline zmm stores with AVX512 enabled. Scalar row-pointer/frame reloads and address multiplies remain and are retained as costs. Input body sizes grow to 927/1047 bytes from 773/768. Caller code can differ through ordinary PGO despite byte-exact non-transform source; all tiers are retained. Source finite/alias/range guards, weight preparation, reduction, inverse transform and epilogue are unchanged and all captured records remain exact. No timing or product benefit is inferred.')
    OUT.mkdir();(OUT/'review.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    (REPORT/'codegen-observations-20260923.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(review=pin(OUT/'review.json'),bodies=len(all_bodies),optimized_reductions=len(reviews))))

if __name__=='__main__':main()
