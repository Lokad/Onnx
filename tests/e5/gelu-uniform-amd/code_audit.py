"""Verify the observed optimized branch really bypasses sixteen vector FMAs and exp reconstruction."""
import re

def inspect(text):
    sections=re.split(r'(?=; Assembly listing for method )',text)
    def method(name):
        matches=[s for s in sections if s.startswith('; Assembly listing for method Kernels:'+name+'(')]
        assert len(matches)==1 and '(FullOpts)' in matches[0].splitlines()[0]
        return matches[0]
    control=method('CopyA');candidate=method('Conditional')
    chunks=re.split(r'(?=^G_M\d+_IG\d+:)',candidate,flags=re.M)
    blocks={re.match(r'(G_M\d+_IG\d+):',s)[1]:s.split('\nRWD00')[0] for s in chunks if re.match(r'G_M\d+_IG\d+:',s)}
    labels=list(blocks)
    matches=[]
    for label,block in blocks.items():
        hit=re.search(r'vptest\s+(ymm\d+),\s*\1\s*\n\s*je\s+(?:SHORT\s+)?(G_M\d+_IG\d+)\b',block)
        if hit:matches.append((label,hit[1],hit[2]))
    assert len(matches)==1
    branch,mask,fast=matches[0];prefix=blocks[branch];fast_body=blocks[fast]
    assert re.search(r'vcmpgtps\s+'+mask+r',',prefix) and '3F6C00003F6C0000h' in candidate
    assert len(re.findall(r'\bvfmadd\w*\b',prefix))==6
    assert re.search(r'\bvorps\b',fast_body) and not re.search(r'\b(?:call|vfmadd\w*|j\w+)\b',fast_body)
    join=labels[labels.index(fast)+1];store=blocks[join]
    assert 'vaddps' in store and 'vmulps' in store and re.search(r'vmovups\s+ymmword ptr',store)
    slow=labels[labels.index(branch)+1];slow_body=blocks[slow]
    assert len(re.findall(r'\bvfmadd\w*\b',slow_body))==16
    assert 'vcvttps2dq' in slow_body and 'vpslld' in slow_body and not re.search(r'\bcall\b',slow_body)
    assert re.search(r'\bjmp\s+(?:SHORT\s+)?'+join+r'\b',slow_body)
    assert len(re.findall(r'\bvfmadd\w*\b',control))==22
    return dict(passed=True,control_code_bytes=int(re.search(r'Total bytes of code (\d+)',control)[1]),candidate_code_bytes=int(re.search(r'Total bytes of code (\d+)',candidate)[1]),
                branch_block=branch,predicate_register=mask,fast_block=fast,slow_block=slow,join_block=join,small_fma=6,skipped_fma=16,
                prefix=prefix,fast_path=fast_body+store,skipped_path=slow_body)
