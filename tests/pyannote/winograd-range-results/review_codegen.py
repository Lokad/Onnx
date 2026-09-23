"""Review all emitted optimized reductions; retain every raw tier."""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/pyannote-winograd-range-codegen-amd-20260923'
OUT=ROOT/'artifacts/pyannote-winograd-range-codegen-review-20260923'
REPORT=Path(__file__).resolve().parent

def pin(p):return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(p):return json.loads(p.read_text(encoding='utf8'))

def main():
    assert not OUT.exists()
    assert pin(BASE/'closed.json')['sha256']=='2e4a59d5040346eec9ca74076be854aa8bd1863c0460ab7f4cac12098f89e2f8'
    for name,wanted in read(BASE/'closed.json')['files'].items():assert pin(BASE/name)==wanted,name
    code=read(BASE/'listings.json');reviews=[];all_bodies=[];inputs=[];guards=[]
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
        wrapper,=[b for b in bodies if ':TransformWinogradInput(' in b['method']]
        contiguous,=[b for b in bodies if ':TransformWinogradInputContiguous(' in b['method']]
        assert wrapper['tier']==contiguous['tier']=='Tier0-FullOpts'
        body=contiguous['body'];channel=body[body.index('G_M000_IG03:'):body.index('G_M000_IG08:')]
        assert not re.search(r'\bidiv\b|\bvgather|\bvfmadd|^\s+call\s',channel,re.M)
        assert len(re.findall(r'\bvpermps\b',channel))==4
        assert re.findall(r'\bvpermps\s+ymm\d+, ymm0, ymmword ptr \[r(?:10|13)([^\]]*)\]',channel)==['','+0x20','+0x08','+0x28']
        assert re.findall(r'\bvperm2f128[^\n]+, (\d+)',channel)==['32','49','32','49']
        assert len(re.findall(r'\bvaddps\b',channel))==2 and len(re.findall(r'\bvsubps\b',channel))==6
        assert '0000000200000000h, 0000000600000004h, 0000000300000001h, 0000000700000005h' in body
        for label in ['04','06','03']:assert re.search(r'jl\s+G_M000_IG'+label,channel)
        assert len(re.findall(r'\bsub\s+rsp, 512',body))==1
        assert body.count('CORINFO_HELP_MEMZERO')==(1 if width==256 else 0)
        assert len(re.findall(r'\bvmovdqu32\s+zmmword ptr',body))==(8 if width==512 else 0)
        w=wrapper['body'];first_stack=re.search(r'\bsub\s+rsp, 512',w).start();dispatch=w[:first_stack]
        assert len(re.findall(r'\bidiv\b',dispatch))==3 and len(re.findall(r'\bidiv\b',w))==5
        assert '+0x03]' in dispatch and '+0x11]' in dispatch and re.search(r'cmp\s+\w+, 8',dispatch)
        target='G_M000_IG'+('41' if width==256 else '35')
        assert re.search(r'jl\s+'+target,dispatch)
        routed=w[w.index(target+':'):]
        assert routed.count('call     [Lokad.Onnx.ConvBlockedSpatial:TransformWinogradInputContiguous(')==1
        assert 'CORINFO_HELP_MEMZERO' not in routed and not re.search(r'\bsub\s+rsp, 512',routed)
        assert len(re.findall(r'\bvgatherdps\b',w))==4
        inputs.append(dict(width=width,helper_tier=contiguous['tier'],helper_bytes=contiguous['code_bytes'],
            wrapper_bytes=wrapper['code_bytes'],body_sha256=contiguous['body_sha256'],wrapper_sha256=wrapper['body_sha256'],
            interior_dispatch_divisions=3,interior_dispatch_skips_fallback_stack=True,
            vector_memory_operands_per_channel=16,element_permutations_per_channel=16,
            half_permutations_per_channel=16,vector_adds_per_channel=8,vector_subtracts_per_channel=24,
            contiguous_stack_storage=512,stack_initialization_helpers=1 if width==256 else 0,
            inline_stack_zero_stores=8 if width==512 else 0,channel_loop_calls=0,
            channel_frame_references=[line.strip() for line in channel.splitlines() if re.search(r'\[(?:rbp|rsp)[+\-]',line)],
            fallback_masked_gathers_per_channel=16))
        range_bodies=[b for b in bodies if ':EpilogueRange(' in b['method']]
        optimized=[b for b in range_bodies if b['tier'].startswith('Tier1')]
        assert len(range_bodies)==5 and len(optimized)==3
        for row in range_bodies:
            assert '7E7FFFFFh' in row['body'] and '7FFFFFFF' in row['body']
        for row in optimized:
            body=row['body'];loop=body[body.index('G_M000_IG03:'):body.index('G_M000_IG05:')]
            assert not re.search(r'\[(?:rbp|rsp)[+\-]|^\s+call\s|\bimul\b|\bidiv\b',loop,re.M)
            assert len(re.findall(r'\bvpandd?\s+'+reg+r'2, '+reg+r'0, '+reg+r'word ptr',loop))==1
            assert len(re.findall(r'\bvcmpgtps\b',loop))==1
            assert re.search(r'\badd\s+eax, '+str(width//32)+r'\b',loop)
            assert re.search(r'jle\s+SHORT G_M000_IG03',loop)
            assert re.search(r'\blea\s+\w+, \[\w+-0x'+('08' if width==256 else '10')+r'\]',body[:body.index('G_M000_IG03:')])
            assert len(re.findall(r'\bvmovmskps\b',loop))==(1 if width==256 else 0)
            assert len(re.findall(r'\bkmovw\b',loop))==(1 if width==512 else 0)
            target,=re.findall(r'jne\s+SHORT (G_M000_IG\d+)',loop)
            rejected=body[body.index(target+':'):];assert re.search(r'\bxor\s+eax, eax',rejected.split('ret')[0])
            tail=body[body.index('G_M000_IG05:'):]
            assert 'vucomiss' in tail and 'vandps' in tail and re.search(r'ja\s+SHORT '+target,tail)
            assert 'CORINFO_HELP_RNGCHKFAIL' in tail
            guards.append(dict(width=width,tier=row['tier'],bytes=row['code_bytes'],lanes=width//32,
                limit_bits='7e7fffff',absolute_mask_bits='7fffffff',vector_loop_frame_references=0,
                vector_loop_calls=0,vector_loop_loads=1,comparison='strict ordered greater-than',
                mask_extraction='vmovmskps' if width==256 else 'kmovw',
                load_bound='index <= length - lanes',scalar_tail=True,body_sha256=row['body_sha256']))
        for index,row in enumerate(bodies):
            assert row['complete_body'] and row['complete_uninterleaved'] and row['body']==row['raw_body']
            name=f'{role}-{index:02}.asm';assert not (REPORT/name).exists()
            (REPORT/name).write_text(row['raw_body'],encoding='utf8',newline='\n')
            all_bodies.append(dict(role=role,file=name,method=row['method'],tier=row['tier'],bytes=row['code_bytes'],digest=pin(REPORT/name)))
    value=dict(passed=True,closure=pin(BASE/'closed.json'),reviewer=pin(Path(__file__)),reductions=reviews,
        range_guards=guards,input_transforms=inputs,all_bodies=all_bodies,all_tiers_retained=True,no_performance_measurement=True,
        manual_review='Both optimized contiguous helpers execute through a geometry-guarded branch before fallback stack allocation. Three divisions select a full same-row eight-tile interior group; top+3<h and left+17<w guard all memory operands. Four contiguous loads are folded into vpermps at byte offsets 0,32,8,40. The constant orders lanes 0,2,4,6,1,3,5,7; vperm2f128 controls32/49 combine lower/upper halves. Each four-row loop uses four memory permutations and four half combinations; explicit ordered transforms retain 8 vector adds and24 subtracts per channel. There are no gathers, divisions or calls in contiguous channel processing. A512-byte row array is still cleared per interior batch: one MEMZERO helper in256mode or eightzmmstores in512mode. The256helper spills/reloads its constant across MEMZERO and reloads once per row;512keeps it in a register. Scalar pointer work and explicit row stores/loads remain. Helpers are598bytes each; dispatch-plus-masked bodies1153/1170bytes. The original masked algorithm remains for borders/tails and pays additional dispatch costs. All six optimized reductions retain eight ordered channel FMAs, one shared weight load, eight accumulators, and no loop frame references/spills/calls. Inverse transforms and all finite/alias/range guards remain. All56emitted bodies are retained. All six optimized range bodies use sign-mask0x7fffffff and unchanged threshold0x7e7fffff, one vector memory operand, strict greater-than comparison and any-lane rejection. They advance8/16floats under index<=length-lanes, have no calls/frame references inside the vector loop, and keep the scalar absolute-value/comparison tail and range-failure helper. OSR vector imports are outside the loop. FullTier1range sizes are209/205bytes. Existing finite scans and every rejection contract remain;10566dedicated guard cases perwidth passed before this capture. Earlier diagnostic filters omitted the predecessor range helper, so this capture does not claim to prove its generated loop was scalar. No application or native speed claim is inferred.')
    OUT.mkdir();(OUT/'review.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    (REPORT/'codegen-observations-20260923.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(review=pin(OUT/'review.json'),bodies=len(all_bodies),optimized_reductions=len(reviews))))

if __name__=='__main__':main()
