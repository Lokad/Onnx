"""Inspect actual AMD FullOpts loops, including literal input displacements."""
import re

def inspect(text,owner):
    sections=re.split(r'(?=; Assembly listing for method )',text)
    matches=[s for s in sections if s.startswith('; Assembly listing for method WidthProbe.'+owner+':PackedTile12(')]
    assert len(matches)==1 and '(FullOpts)' in matches[0].splitlines()[0]
    section=matches[0];lines=section.splitlines();labels={}
    for i,line in enumerate(lines):
        m=re.match(r'(G_M\d+_IG\d+):',line)
        if m:labels[m[1]]=i
    loops=[]
    for i,line in enumerate(lines):
        m=re.search(r'\bj[a-z]+\s+(?:SHORT\s+)?(G_M\d+_IG\d+)\b',line)
        if m and m[1] in labels and labels[m[1]]<i:
            start=labels[m[1]];body='\n'.join(lines[start:i+1])
            if len(re.findall(r'\bvfmadd\w*\b',body))==24:loops.append((i-start,start,i,body))
    assert loops
    _,start,end,hot=min(loops)
    stack=lambda l:bool(re.search(r'\[(?:rbp|rsp)(?:\+|\-|\])',l))
    return dict(method=owner+'.PackedTile12',code_bytes=int(re.search(r'Total bytes of code (\d+)',section)[1]),
        hot_loop=hot,fma=len(re.findall(r'\bvfmadd\w*\b',hot)),broadcasts=len(re.findall(r'\bvbroadcastss\b',hot)),
        movsxd=len(re.findall(r'\bmovsxd\b',hot)),lea=len(re.findall(r'\blea\b',hot)),
        hot_stack=[l.strip() for l in hot.splitlines() if stack(l)],
        vector_stack=[l.strip() for l in lines if stack(l) and re.search(r'\b[xyz]mm(?:word|\d+)\b',l)],
        calls=[l.strip() for l in hot.splitlines() if re.search(r'\bcall\b',l)])

def pointer_gate(row):
    assert row['fma']==24 and row['broadcasts']==12 and not row['vector_stack'] and not row['calls'] and not row['hot_stack']
    assert row['movsxd']<=2 and row['lea']<=2
    loads=re.findall(r'\bvbroadcastss\s+zmm\d+, dword ptr \[(r\w+)(?:\+(0x[0-9A-Fa-f]+|\d+))?\]',row['hot_loop'])
    assert len(loads)==12 and len({base for base,offset in loads})==1
    assert [int(offset,0) if offset else 0 for base,offset in loads]==list(range(0,48,4))
    base=loads[0][0]
    assert re.search(r'\badd\s+'+re.escape(base)+r', (?:48|0x30)\b',row['hot_loop'])
    return dict(passed=True,base_register=base,byte_offsets=list(range(0,48,4)),advance_bytes=48)

def audit(text):
    rows=[inspect(text,owner) for owner in ['Kernels','InputPacked','InputPointer']]
    return dict(passed=True,methods=rows,pointer=pointer_gate(rows[-1]))
