"""Reconcile both actual four-block reductions and retain every emitted tier."""
from collections import Counter
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/pyannote-convolution-four-codegen-amd-v2-20260923'
OUT=ROOT/'artifacts/pyannote-convolution-four-codegen-review-20260923'
REPORT=ROOT/'tests/pyannote/convolution-four-results'

def pin(p):return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(p):return json.loads(p.read_text(encoding='utf8'))

def main():
    assert not OUT.exists() and not REPORT.exists()
    assert pin(BASE/'closed.json')['sha256']=='36c548cf1c577df34221ece455f80143c6e0fa46f60522e51e204b28ee14f169'
    for name,wanted in read(BASE/'closed.json')['files'].items():assert pin(BASE/name)==wanted,name
    code=read(BASE/'listings.json');rows=[]
    candidates=[r for r in code['candidate'] if ':Kernel512Four(' in r['method'] and r['tier'].startswith('Tier1')]
    assert [r['code_bytes'] for r in candidates]==[[4309],[4346]]
    for row in candidates:
        assert row['complete_body'] and row['complete_uninterleaved'] and row['raw_body']==row['body']
        reduction=[b for b in row['reductions'] if b['fma_instructions']>4]
        assert [b['label'] for b in reduction]==[f'G_M000_IG{i:02}' for i in range(10,17)]
        body=''.join(b['body'] for b in reduction)
        fmas=re.findall(r'\bvfmadd231ps\s+(zmm\d+), (zmm\d+), (zmm\d+)',body)
        assert len(fmas)==216 and Counter(a for a,_,_ in fmas)==Counter({f'zmm{i}':9 for i in range(24)})
        assert fmas==fmas[:24]*9
        assert fmas[:24]==[(f'zmm{i}',f'zmm{24+i%4}','zmm28') for i in range(24)]
        loads=re.findall(r'\bvmovups\s+(zmm2[4-7]), zmmword ptr \[(\w+)\]',body)
        assert len(loads)==36 and loads==loads[:4]*9
        assert loads[:4]==[('zmm24','r10'),('zmm25','r11'),('zmm26','rbx'),('zmm27','r15')]
        assert re.findall(r'\badd\s+(\w+), 64\b',body)==[p for _,p in loads[:4]]*9
        broadcasts=re.findall(r'\bvbroadcastss\s+zmm28, dword ptr \[([^\]]+)\]',body)
        expected=[base+suffix for base in ['r8','r13','r8','rdx','r8','rdx','rdi','rdx','rdi']
                  for suffix in ['', '+4*rcx','+4*r9','+4*rsi','+4*r12','+4*rax']]
        assert broadcasts==expected and sum(b['input_broadcasts'] for b in reduction)==54
        assert not re.search(r'^\s+(?:j\w+|call)\s',body,re.M)
        assert not any(b['vector_stack_references'] for b in reduction)
        assert sum(b['integer_multiplies'] for b in reduction)==1
        assert sum(b['arithmetic_shifts'] for b in reduction)==1
        assert re.search(r'\band\s+r9d, 15\b',body)
        after_entry=row['body'][row['body'].index('G_M000_IG02:'):]
        assert not [line for line in after_entry.splitlines() if re.search(r'\b[xyz]mm\d+\b',line) and re.search(r'\[(?:rbp|rsp)',line)]
        assert not re.search(r'^\s+call\s',row['body'],re.M)
        assert len(re.findall(r'\bvmulps\b',row['body']))==len(re.findall(r'\bvaddps\b',row['body']))==4
        assert 'jmp      G_M000_IG10' in row['body']
        rows.append(dict(tier=row['tier'],code_bytes=row['code_bytes'],blocks=[b['label'] for b in reduction],
            ordered_fma_group=fmas[:24],groups=9,fmas=216,input_broadcasts=54,weight_loads=36,
            four_weight_advances=9,integer_multiplies=1,arithmetic_shifts=1,
            scalar_stack_references=sum(len(b['scalar_stack_references']) for b in reduction),
            vector_loop_stack_references=0,separate_tail_mul_add_instructions=4,
            raw_sha256=row['raw_sha256'],body_sha256=row['body_sha256']))
    control=next(r for r in code['production'] if r['code_bytes']==[2176])
    fallback=next(r for r in code['candidate'] if ':Kernel512(' in r['method'] and r['tier'].startswith('Tier1'))
    assert control['body']==fallback['body']
    value=dict(passed=True,closure=pin(BASE/'closed.json'),listings=pin(BASE/'listings.json'),reviewer=pin(Path(__file__)),
        candidate=rows,fallback_optimized_body_exact=True,fallback_body_sha256=fallback['body_sha256'],
        all_tiers_retained=True,no_performance_measurement=True,execute_native_capture=False,
        manual_review='Both optimized four-block bodies derive row0 from channel>>4, blockStride and channel&15, then row1/row2 from successive rowStride additions. Nine straight-line spatial steps preserve 24 ordered FMA accumulators and four weight advances. The channel backedge is outside the branch-free seven-block reduction. Scalar frame references remain (10/11); there are no vector spills after OSR entry or inner calls. OSR entry imports the 24 accumulators from the instrumented frame. Stores cover all four full output blocks. Tail row/column loops preserve four separate multiply/add instructions beyond fusedEnd and four fused operations before it. The 2,176-byte original-kernel optimized body is text-identical to current. The later original-kernel entry emitted only instrumented Tier0 in this candidate capture; all bodies remain. Execute source/IL was checked by the normal build; no native Execute listing is claimed. Code inspection does not establish speed.')
    OUT.mkdir();(OUT/'review.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    REPORT.mkdir()
    for role,bodies in code.items():
        for i,row in enumerate(bodies):
            stem=f'{role}-{i:02}'
            (REPORT/(stem+'.raw.asm')).write_text(row['raw_body'],encoding='utf8')
            (REPORT/(stem+'.asm')).write_text(row['body'],encoding='utf8')
    (REPORT/'codegen-observations-20260923.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(review=pin(OUT/'review.json'),candidate=rows,fallback_body_exact=True)))

if __name__=='__main__':main()
