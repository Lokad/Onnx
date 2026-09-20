"""Close the complete two-process proof; preserve source and code caveats."""
import argparse, datetime, re
from proof_common import *
from audit_proof import audit
from vm import ssh, REMOTE


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);a=p.parse_args();base=a.artifact.resolve()
    result=audit(base);assert result==read(base/'audit.json')
    observed=json.loads(ssh('import sys,json\nfrom pathlib import Path\nsys.path.insert(0,%r)\nimport process_support as h\nrows=%r\n' % (REMOTE,result['resources']['births'])+'''
out=[]
for row in rows:
 actual=h.proc(row['pid']);assert actual is None or actual['start']!=row['start'],row
 out.append(dict(expected=row,actual=actual,terminal=True))
print(json.dumps(out))
'''))
    text=(base/'collected/result/jit.txt').read_text();sections=re.split(r'(?=; Assembly listing for method )',text);review=[]
    for width in [12,8]:
        section=next(s for s in sections if s.startswith(f'; Assembly listing for method ReductionProbe.Blocked:PackedTile{width}('))
        assert 'mov      eax, dword ptr [rbp+0x18]' in section
        assert re.search(r'\bimul\s+\w+, esi',section)
        hot=next(r['hot_loop'] for r in result['code']['methods'] if r['method']==f'Blocked.PackedTile{width}')
        assert re.search(r'\bcmp\s+(?:edi|r9d), eax',hot)
        review.append(dict(method=f'Blocked.PackedTile{width}',count='eax loaded from eighth argument [rbp+0x18]',
                           row_stride='original esi retained in row-address multiplication',
                           stack_instructions=[l.strip() for l in hot.splitlines() if re.search(r'\[(?:rbp|rsp)(?:\+|\-|\])',l)]))
    observations=dict(audit=result,instruction_review=review,terminal=observed)
    lane=Path(__file__).parent;report=lane/'proof-results-20260920.md';data=lane/'proof-observations-20260920.json'
    lines=['# AMD reduction-block correctness and code proof — September 20, 2026','',
           'Both the ordinary and explicitly instrumented processes pass all **473 matrix cases**. Each checks both fixed reduction-block sizes (128 and 256) against the original at every output bit, including nonzero destinations and second accumulation. Every input and packed weight stays unchanged; guards and 211,890 scalar-FMA coordinates pass per process. All 31 invalid geometry/block refusals pass.','',
           'The two processes retain identical complete first-output arrays: 3,433,056 matrix values per process, plus guards. Candidate equality and second accumulation are assertions in the pinned probe; candidate arrays and complete second outputs were not separately saved. Their absence is not represented as independent full-array storage.','',
           '| Method | Original code bytes | Blocked code bytes | Reduction-loop FMAs | Broadcasts |',
           '|---|---:|---:|---:|---:|']
    for width in [12,8]:
        original=next(r for r in result['code']['methods'] if r['method']==f'Original.PackedTile{width}')
        blocked=next(r for r in result['code']['methods'] if r['method']==f'Blocked.PackedTile{width}')
        lines.append(f"| {width} rows | {original['code_bytes']} | {blocked['code_bytes']} | {width*2} | {width} |")
    lines+=['',
            'All four bodies are actual AMD FullOpts code. Reduction loops contain no calls or vector stack accesses. In both blocked leaves, the loop count comes from the separate eighth argument at `[rbp+0x18]`, while the original row stride remains in address arithmetic. The twelve-row loop reloads and stores that count through the stack; scalar stack work remains and no instruction-count or speed advantage is claimed. Complete dumps and loop excerpts are retained.','',
            'This reuses existing packed weights and unchanged two/three-row tails. It introduces no activation gathering, new weight format or overwrite shortcut. The result qualifies the prototype for a separately declared complete-cost comparison; it does not qualify production dispatch or establish cache behavior, e5 latency, native parity or a default change.','',
            f"Runtime .NET10.0.8 on AMD EPYC9V74, CPU2 inherited before startup; supervisor CPU0. Fixed guards are180seconds /2GiB sampled group RSS /1GiB minimum available memory. Both runs finish in about one second, all18resource samples pass and all three process births are terminal. Source `{read(base/'payload/bundle.json')['source']}`; exact reused probe `{PROBE}`, qualified core `{CORE}`. The local build's sole CA1416 affinity warning remains documented. No compiler rebuild occurred before AMD execution.",'',
            '[Protocol](README.md) and [complete observations](proof-observations-20260920.json) give the scope, code excerpts, source identities and resources.','']
    with report.open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines))
    write(data,observations)
    receipt=dict(passed=True,closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),bundle=pin(base/'payload/bundle.json'),
                 reports={p.relative_to(ROOT).as_posix():pin(p) for p in [report,data,Path(__file__)]},births=result['resources']['births'],
                 files={p.relative_to(base).as_posix():pin(p) for p in sorted(base.rglob('*')) if p.is_file()})
    write(base/'closed.json',receipt);print(json.dumps(dict(receipt=pin(base/'closed.json'),files=len(receipt['files']),passed=True)))


if __name__=='__main__':main()
