"""Record actual FullOpts reduction loops without inventing cache measurements."""
import hashlib, re


def inspect(text):
    sections=re.split(r'(?=; Assembly listing for method )',text);rows=[]
    for owner in ['Original','Blocked']:
        for width in [12,8]:
            matches=[s for s in sections if s.startswith(f'; Assembly listing for method ReductionProbe.{owner}:PackedTile{width}(')]
            assert len(matches)==1 and '(FullOpts)' in matches[0].splitlines()[0]
            body=matches[0];lines=body.splitlines();labels={m[1]:i for i,line in enumerate(lines) if (m:=re.match(r'(G_M\d+_IG\d+):',line))}
            loops=[]
            for i,line in enumerate(lines):
                branch=re.search(r'\bj[a-z]+\s+(?:SHORT\s+)?(G_M\d+_IG\d+)\b',line)
                if branch and branch[1] in labels and labels[branch[1]]<i:
                    start=labels[branch[1]];hot='\n'.join(lines[start:i+1])
                    if len(re.findall(r'\bvfmadd\w*\b',hot))==width*2:loops.append((i-start,hot))
            assert loops;hot=min(loops)[1]
            assert not re.search(r'\bcall\b',hot)
            assert not any(re.search(r'\[(?:rsp|rbp)(?:\+|\-|\])',line) and re.search(r'\b[xyz]mm(?:word|\d+)\b',line) for line in hot.splitlines())
            assert len(re.findall(r'\bvbroadcastss\b',hot))==width
            rows.append(dict(method=f'{owner}.PackedTile{width}',header=lines[0],code_bytes=int(re.search(r'Total bytes of code (\d+)',body)[1]),
                             sha256=hashlib.sha256(body.encode()).hexdigest(),hot_loop=hot,fma=width*2,broadcasts=width))
    return dict(passed=True,methods=rows)
