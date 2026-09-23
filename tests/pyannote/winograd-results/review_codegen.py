"""Review all emitted optimized reductions; retain every raw tier."""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/pyannote-winograd-codegen-amd-20260923'
OUT=ROOT/'artifacts/pyannote-winograd-codegen-review-20260923'
REPORT=Path(__file__).resolve().parent

def pin(p):return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(p):return json.loads(p.read_text(encoding='utf8'))

def main():
    assert not OUT.exists()
    assert pin(BASE/'closed.json')['sha256']=='28c8d94684024251f515d5902757056ae59aff9824a35df5886a589024446566'
    for name,wanted in read(BASE/'closed.json')['files'].items():assert pin(BASE/name)==wanted,name
    code=read(BASE/'listings.json');reviews=[];all_bodies=[]
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
        for index,row in enumerate(bodies):
            assert row['complete_body'] and row['complete_uninterleaved'] and row['body']==row['raw_body']
            name=f'{role}-{index:02}.asm';assert not (REPORT/name).exists()
            (REPORT/name).write_text(row['raw_body'],encoding='utf8',newline='\n')
            all_bodies.append(dict(role=role,file=name,method=row['method'],tier=row['tier'],bytes=row['code_bytes'],digest=pin(REPORT/name)))
    value=dict(passed=True,closure=pin(BASE/'closed.json'),reviewer=pin(Path(__file__)),reductions=reviews,
        all_bodies=all_bodies,all_tiers_retained=True,no_performance_measurement=True,
        manual_review='All six optimized reduction bodies visit channels in increasing order with eight independent accumulators, one shared weight load and eight inputs, without vector spills, scalar frame references or calls in the reduction. AVX512 folds broadcasts into FMA memory operands; AVX256 uses separate broadcasts. OSR entries import accumulators from their instrumented frames; those loads are retained. Full Tier1 inverse transforms have 12 vector adds and 12 subtracts, guarded odd-row/column stores, two integer divisions per tile/channel block and scalar address work. Input transforms retain scalar loops, stack tile storage, boundary checks, two divisions per tile/channel, scratch clearing and range checks. Weight transforms use explicit scalar adds/subtracts and multiply-by-half. Execute retains finite scans, alias/extent validation, all scratch guards and epilogue range scans. These are costs in the prospective whole-call screen, not excluded setup. No native speed or product dispatch is inferred.')
    OUT.mkdir();(OUT/'review.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    (REPORT/'codegen-observations-20260923.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(review=pin(OUT/'review.json'),bodies=len(all_bodies),optimized_reductions=len(reviews))))

if __name__=='__main__':main()
