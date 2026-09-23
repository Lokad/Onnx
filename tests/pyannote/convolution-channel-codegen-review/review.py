"""Review the actual channel-block kernel; retain raw text and an explicit stdout repair."""
from collections import Counter
import hashlib,json,re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/pyannote-convolution-channel-codegen-amd-20260923'
OUT=ROOT/'artifacts/pyannote-convolution-channel-codegen-review-20260923'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def pin(p):return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf8')
def blocks(body):
    starts=list(re.finditer(r'^(G_M\d+_IG\d+):[^\n]*\n',body,re.M))
    return {s.group(1):body[s.end():starts[i+1].start() if i+1<len(starts) else len(body)] for i,s in enumerate(starts)}

def main():
    assert not OUT.exists()
    assert pin(BASE/'closed.json')['sha256']=='d0cb1cb7791709e20a7d07901119dee6afb82362e293615c072133b71cb42adb'
    for name,wanted in read(BASE/'closed.json')['files'].items():assert pin(BASE/name)==wanted,name
    code=read(BASE/'listings.json');repairs=[];normalized={}
    marker='Diagnostic complete: 432 exact ordinary graph outputs.\n'
    for role,rows in code.items():
        normalized[role]=[]
        for row in rows:
            raw=row['body'];body=raw.replace(marker,'')
            # Managed stdout can split an instruction; removing only this exact,
            # independently known complete message rejoins its original bytes.
            assert raw.count(marker)<=1
            if marker in raw:repairs.append(dict(role=role,tier=row['tier'],offset=raw.index(marker),removed=marker,
                raw_sha256=hashlib.sha256(raw.encode()).hexdigest(),rejoined_sha256=hashlib.sha256(body.encode()).hexdigest()))
            assert 'Diagnostic' not in body and body.count('; Assembly listing for method ')==1
            assert len(re.findall(r'; Total bytes of code \d+',body))==1
            labels=list(blocks(body));assert len(labels)==len(set(labels))
            assert set(re.findall(r'\bG_M\d+_IG\d+\b',body))==set(labels)
            normalized[role].append(dict(row,body=body,managed_stdout_rejoined=marker in raw))
    a,=[r for r in normalized['production'] if r['tier'].startswith('Tier1')]
    b,=[r for r in normalized['candidate'] if r['tier'].startswith('Tier1')]
    assert a['code_bytes']==[2176] and b['code_bytes']==[3680]
    ab,bb=blocks(a['body']),blocks(b['body'])
    step=bb['G_M000_IG08'];original=ab['G_M000_IG08']
    # Stack layout differs; every reduction instruction otherwise remains exact.
    def normalize_stack(s):return re.sub(r'\[rbp\+0x[0-9A-F]+\]','[rbp+OFFSET]',s)
    assert normalize_stack(step)==normalize_stack(original)
    assert Counter(re.findall(r'vfmadd231ps\s+(zmm\d+),',step))==Counter({f'zmm{i}':1 for i in range(12)})
    assert step.count('vbroadcastss')==6 and step.count('vmovups')==2
    stack=[(label,line.strip()) for label,part in bb.items() for line in part.splitlines()
           if re.search(r'\b[xyz]mm\d+\b',line) and re.search(r'\[(?:rbp|rsp)',line)]
    assert len(stack)==12 and all(label=='G_M000_IG01' for label,line in stack)
    assert 'cmp      r10d, r11d' in bb['G_M000_IG04'] and 'jge      G_M000_IG12' in bb['G_M000_IG04']
    assert 'imul     r8d, r15d, 144' in bb['G_M000_IG51']
    assert 'mov      r10d, ebx' in bb['G_M000_IG51'] and 'mov      ebx, r15d' in bb['G_M000_IG51']
    assert 'cmp      ebx, 64' in bb['G_M000_IG81'] and 'mov      r11d, 16' in bb['G_M000_IG87']
    assert 'add      r15d, r11d' in bb['G_M000_IG83'] and 'inc      r9d' in bb['G_M000_IG116']
    for zero,load,second,guard,second_load,acc,acc2 in [
        (28,88,30,89,90,0,6),(32,92,34,93,94,1,7),(36,96,38,97,98,2,8),
        (40,100,42,101,102,3,9),(44,104,46,105,106,4,10),(48,108,50,109,110,5,11)]:
        block=lambda n:bb[f'G_M000_IG{n:02d}']
        assert f'vxorps   ymm{acc}, ymm{acc}, ymm{acc}' in block(zero)
        assert f'vmovups  zmm{acc}, zmmword ptr [rdx+' in block(load)
        assert f'vxorps   ymm{acc2}, ymm{acc2}, ymm{acc2}' in block(second)
        assert 'lea      r8d, [r10+0x10]' in block(guard) and 'cmp      r8d, edi' in block(guard) and 'jge' in block(guard)
        assert f'vmovups  zmm{acc2}, zmmword ptr [rdx+' in block(second_load)
        assert 'test     r15d, r15d' in block(zero-1)
    for n in [54,55,73]:
        part=bb[f'G_M000_IG{n}'];assert part.count('vmulps')==2 and part.count('vaddps')==2 and 'vfmadd' not in part
    assert 'vmovups  zmm0, zmmword ptr [rdx+' in bb['G_M000_IG66']
    assert 'cmp      esi, edi' in bb['G_M000_IG68'] and 'jge' in bb['G_M000_IG68']
    assert 'vmovups  zmm6, zmmword ptr [rdx+' in bb['G_M000_IG69']
    # Instruction splitting is visible in the raw capture and not hidden by the
    # older parser's overly broad complete_uninterleaved field.
    assert len(repairs)==2 and all(r['tier']=='Tier1-OSR' for r in repairs)
    OUT.mkdir();save(OUT/'normalized-listings.json',normalized)
    value=dict(passed=True,closure=pin(BASE/'closed.json'),listings=pin(BASE/'listings.json'),reviewer=pin(Path(__file__)),
        normalized_listings=pin(OUT/'normalized-listings.json'),stdout_repairs=repairs,
        original_parser_limitation='complete_uninterleaved only counted code-size footers and did not detect managed stdout interleaved within an instruction. Raw capture remains immutable. Remove only the exact complete driver completion message; no generated instruction is omitted.',
        current_optimized_bytes=2176,candidate_optimized_bytes=3680,current_tier0_bytes=3756,candidate_tier0_bytes=5669,
        reduction_instructions_exact_except_stack_offsets=True,fmas_per_spatial_step=12,broadcasts_per_step=6,weight_loads_per_step=2,
        optimized_vector_stack_references=stack,vector_spills_in_loop=0,
        manual_review='IG27-50 choose zero accumulators for firstChannel=0; IG88-110 load partial output on subsequent blocks, with oc+16<m guards for every second output vector. IG51 offsets weights by firstChannel*144 floats and starts ic at firstChannel. IG04 stops at endChannel; IG81-87/115 choose16 channels at c>=64 and finish all spatial tiles before next channel block. IG65-69 preserve single-position initialization/guards. IG54/55/73 keep separate multiply/add beyond fusedEnd. IG08 reduction instructions match current except frame offsets. Twelve entry loads import OSR state; no vector stack references remain in the optimized loop. Tier0 and every emitted Tier1 body are retained. Kernel256 compiled equivalence is established by build inventory, not a new AVX2 code capture.',
        all_tiers_retained=True,no_worker_rerun=True,no_performance_measurement=True)
    save(OUT/'review.json',value);print(json.dumps(dict(review=pin(OUT/'review.json'),repairs=repairs,loop_vector_spills=0)))

if __name__=='__main__':main()
