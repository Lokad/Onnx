"""Check all three optimized nine-step reductions and retain complete raw tiers."""
from collections import Counter
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/pyannote-convolution-pointer-codegen-amd-20260923'
OUT=ROOT/'artifacts/pyannote-convolution-pointer-codegen-review-20260923'
def pin(p):return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(p):return json.loads(p.read_text(encoding='utf8'))

def main():
    assert not OUT.exists()
    assert pin(BASE/'closed.json')['sha256']=='a6d0a3f70916452362a1da5c9f605146dab41426f7303a39fd6e114628d18ed8'
    for name,wanted in read(BASE/'closed.json')['files'].items():assert pin(BASE/name)==wanted,name
    code=read(BASE/'listings.json');rows=[]
    candidates=[r for r in code['candidate'] if r['tier'].startswith('Tier1')]
    assert [r['code_bytes'] for r in candidates]==[[3177],[3066],[3222]]
    for row in candidates:
        assert row['complete_body'] and row['raw_body'].replace('Diagnostic complete: 432 exact ordinary graph outputs.\n','')==row['body']
        reduction=[b for b in row['reductions'] if b['fma_instructions']>2]
        assert len(reduction)==4
        body=''.join(b['body'] for b in reduction)
        fmas=re.findall(r'\bvfmadd231ps\s+(zmm\d+), (zmm\d+), (zmm\d+)',body)
        assert len(fmas)==108 and Counter(a for a,_,_ in fmas)==Counter({f'zmm{i}':9 for i in range(12)})
        assert fmas==fmas[:12]*9 and all(c=='zmm14' for _,_,c in fmas)
        assert [b for _,b,_ in fmas[:12]]==['zmm12','zmm13']*6
        loads=re.findall(r'\bvmovups\s+(zmm1[23]), zmmword ptr \[(r\d+)\]',body)
        assert len(loads)==18 and loads==loads[:2]*9 and loads[0][0]=='zmm12' and loads[1][0]=='zmm13'
        pointers=[p for _,p in loads[:2]]
        advances=re.findall(r'\badd\s+(r\d+), 64\b',body)
        assert advances==pointers*9
        assert sum(b['input_broadcasts'] for b in reduction)==54
        assert not re.search(r'^\s+(?:j\w+|call)\s',body,re.M)
        assert not any(b['vector_stack_references'] for b in reduction)
        assert sum(b['integer_multiplies'] for b in reduction)==1
        assert sum(b['arithmetic_shifts'] for b in reduction)==1
        assert re.search(r'\band\s+\w+, 15\b',body)
        after_entry=row['body'][row['body'].index('G_M000_IG02:'):]
        vector_stack=[line for line in after_entry.splitlines() if re.search(r'\b[xyz]mm\d+\b',line) and re.search(r'\[(?:rbp|rsp)',line)]
        assert not vector_stack
        assert not re.search(r'^\s+call\s',row['body'],re.M)
        tail_count=2 if row['code_bytes']==[3066] else 6
        assert len(re.findall(r'\bvmulps\b',row['body']))==len(re.findall(r'\bvaddps\b',row['body']))==tail_count
        rows.append(dict(tier=row['tier'],code_bytes=row['code_bytes'],blocks=[b['label'] for b in reduction],
            ordered_fma_group=fmas[:12],groups=9,fmas=108,input_broadcasts=54,weight_loads=18,paired_weight_advances=9,
            integer_multiplies=1,arithmetic_shifts=1,scalar_stack_references=sum(len(b['scalar_stack_references']) for b in reduction),
            vector_loop_stack_references=0,separate_tail_mul_add_instructions=tail_count,raw_sha256=row['raw_sha256'],body_sha256=row['body_sha256'],
            managed_stdout_repairs=row['managed_stdout_repairs']))
    value=dict(passed=True,closure=pin(BASE/'closed.json'),listings=pin(BASE/'listings.json'),reviewer=pin(Path(__file__)),
        candidate=rows,current_optimized_sizes=[r['code_bytes'] for r in code['production'] if r['tier'].startswith('Tier1')],
        all_tiers_retained=True,no_performance_measurement=True,
        manual_review='All three optimized bodies derive row0 from channel>>4, blockStride and channel&15, then row1/row2 by successive rowStride additions. Nine full-tile steps preserve twelve ordered FMA accumulators and paired weight increments. The channel backedge remains outside the branch-free four-block region. Optimized loops have no vector stack references; entry loads import OSR state. Stores retain second-output-block guards. The 3066-byte body retains a runtime kx tail loop with two separate mul/add instructions; the other bodies unroll three tail positions and retain six. These are alternate JIT entry/tiering forms, all kept. Kernel256 compiled identity comes from the normal build, not a new AVX2 capture. Smaller code than M23 and reduced integer work do not establish a speedup.')
    OUT.mkdir();(OUT/'review.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(review=pin(OUT/'review.json'),candidate=rows)))

if __name__=='__main__':main()
