"""Review complete current/candidate reductions while retaining the failed capture."""
from pathlib import Path
import hashlib,json,re,shutil

ROOT=Path(__file__).resolve().parents[3];REPORT=Path(__file__).resolve().parent
FIRST=ROOT/'artifacts/pyannote-winograd-output-blocks-codegen-amd-20260923'
FIX=ROOT/'artifacts/pyannote-winograd-output-blocks-codegen-amd-v2-20260923'
OUT=ROOT/'artifacts/pyannote-winograd-output-blocks-codegen-review-20260923'
FIX_CLOSURE='74a30abd4d5c61502c9f737991cf9bdd49a25addcd1c8ea17fbd4b4aea034b9d'


def read(p):return json.loads(p.read_text(encoding='utf8'))
def pin(p):return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def reduction(block,width):
    body=block['body'];count=block['fma_instructions']
    reg=re.search(r'\bvfmadd231ps\s+([yz]mm)\d+',body).group(1)
    vector_width=256 if reg=='ymm' else 512
    assert vector_width<=width and count in [8,16] and (count!=16 or reg=='zmm')
    assert not block['vector_stack_references'] and not block['scalar_stack_references']
    assert not block['integer_multiplies'] and not block['arithmetic_shifts']
    assert not re.search(r'^\s+call\s|\bidiv\b',body,re.M)
    fmas=re.findall(r'\bvfmadd231ps\s+('+reg+r'\d+), ('+reg+r'\d+), ([^\n]+)',body)
    assert len(fmas)==count and len({r[0] for r in fmas})==count
    weights=re.findall(r'\bvmovups\s+('+reg+r'\d+), '+reg+r'word ptr (\[[^\]]+\])',body)
    assert len(weights)==(2 if count==16 else 1)
    broadcasts=re.findall(r'\bvbroadcastss\s+('+reg+r'\d+), dword ptr (\[[^\]]+\])',body)
    if count==16:
        assert [r[1] for r in fmas]==[weights[i%2][0] for i in range(16)]
        assert len(broadcasts)==8 and '{1to16}' not in body
        assert all(fmas[2*i][2]==fmas[2*i+1][2]==broadcasts[i][0] for i in range(8))
        operands=[r[1] for r in broadcasts]
        assert re.fullmatch(r'\[(r\w+)\]',weights[0][1])
        assert weights[1][1]==weights[0][1][:-1]+'+0x40]'
    elif broadcasts:
        assert reg=='ymm'
        assert len(broadcasts)==8 and all(r[1]==weights[0][0] for r in fmas)
        assert [r[2] for r in fmas]==[r[0] for r in broadcasts]
        operands=[r[1] for r in broadcasts]
    else:
        assert not broadcasts and all(r[1]==weights[0][0] for r in fmas)
        suffix=' {1to'+str(vector_width//32)+'}'
        assert all(r[2].startswith('dword ptr ') and r[2].endswith(suffix) for r in fmas)
        operands=[r[2].removeprefix('dword ptr ').removesuffix(suffix) for r in fmas]
    parsed=[re.fullmatch(r'\[(\w+)(?:\+(0x[0-9A-F]+))?\]',v).groups() for v in operands]
    assert len({p[0] for p in parsed})==1
    assert [int(p[1],16) if p[1] else 0 for p in parsed]==list(range(0,32,4))
    assert re.search(r'\badd\s+'+parsed[0][0]+r', 32\b',body)
    counter=re.findall(r'\b(inc|dec)\s+\w+',body);assert len(counter)==1
    if counter==['dec']:assert re.search(r'jne\s+(?:SHORT )?'+block['label']+r'\b',body)
    return dict(label=block['label'],fmas=count,vector_width=vector_width,counter=counter[0],accumulators=[r[0] for r in fmas],
        weight_vectors=len(weights),explicit_broadcasts=len(broadcasts),
        embedded_broadcasts=body.count('{1to'),reduction_loop_frame_references=0,
        reduction_loop_calls=0,input_offsets_bytes=list(range(0,32,4)),input_step_bytes=32)


def main():
    assert not OUT.exists()
    for base,digest,passed in [(FIRST,'116194a103a73afba346d03d6325fe2bf7a011404b4ae905325b884b401512c9',False),(FIX,FIX_CLOSURE,True)]:
        assert pin(base/'closed.json')['sha256']==digest
        closure=read(base/'closed.json');assert closure['passed']==passed
        for name,wanted in closure['files'].items():assert pin(base/name)==wanted,name
    original=read(FIRST/'listings.json');correction=read(FIX/'listings.json')
    selected={};all_bodies=[];raw_files=[];reviews=[];sizes={}
    for origin,base,streams in [('original',FIRST,original),('correction',FIX,correction)]:
        for stream,bodies in streams.items():
            source=base/'collected'/stream/'jit.asm';name=origin+'-'+stream+'.asm'
            assert not (REPORT/name).exists()
            raw_files.append(dict(origin=origin,stream=stream,file=name,pin=pin(source),source=source.relative_to(ROOT).as_posix()))
            for index,b in enumerate(bodies):
                all_bodies.append(dict(origin=origin,stream=stream,index=index,method=b['method'],tier=b['tier'],
                    bytes=b['code_bytes'],line=b['line'],complete=b['complete_body'],raw_sha256=b['raw_sha256']))
    for stream,bodies in original.items():
        role,_,width_text=stream.split('-');width=int(width_text)
        affected={f'Lokad.Onnx.ConvBlockedSpatial:{name}Winograd{width}(' for name in ['Multiply','Output']}
        selected[stream]=[b for b in bodies if not (role=='current' and any(b['method'].startswith(a) for a in affected))]
        if role=='current':
            for kind in ['multiply','output']:selected[stream]+=correction[f'current-{kind}-{width}']
        assert all(b['complete_body'] and b['complete_uninterleaved'] and not b['managed_stdout_repairs'] for b in selected[stream])
        private=[b for b in selected[stream] if b['method'].startswith(f'Lokad.Onnx.ConvBlockedSpatial:MultiplyWinograd{width}(')]
        assert private and any(b['tier']=='Instrumented Tier0' for b in private) and any(b['tier']=='Tier1' for b in private)
        for b in selected[stream]:
            targeted=b['method'].startswith('Lokad.Onnx.ConvBlockedSpatial:MultiplyWinograd') or b['method'].startswith('Lokad.Onnx.ConvBlockedSpatial:ExecuteWinograd(')
            if not targeted or not b['tier'].startswith('Tier1'):continue
            loops=[reduction(r,width) for r in b['reductions']]
            if ':MultiplyWinograd' in b['method']:
                expected=[16,8] if role=='candidate' and width==512 else [8]
                assert [r['fmas'] for r in loops]==expected
                assert not re.search(r'^\s+call\s',b['body'],re.M)
            reviews.append(dict(stream=stream,method=b['method'],tier=b['tier'],bytes=b['code_bytes'],
                body_sha256=b['body_sha256'],loops=loops))
        full,=[b for b in private if b['tier']=='Tier1'];sizes[stream]=dict(multiply=full['code_bytes'][0])
        inverse,=[b for b in selected[stream] if b['method'].startswith(f'Lokad.Onnx.ConvBlockedSpatial:OutputWinograd{width}(') and b['tier']=='Tier1']
        assert len(re.findall(r'\bvaddps\b',inverse['body']))==12
        assert len(re.findall(r'\bvsubps\b',inverse['body']))==12
        assert not re.search(r'\bvfmadd|^\s+call\s',inverse['body'],re.M)
        sizes[stream]['inverse']=inverse['code_bytes'][0]
        caller,=[b for b in selected[stream] if b['method'].startswith('Lokad.Onnx.ConvBlockedSpatial:ExecuteWinograd(') and b['tier']=='Tier1']
        sizes[stream].update(execute=caller['code_bytes'][0],inline_fmas=sum(r['fma_instructions'] for r in caller['reductions']),
            multiply512_calls=caller['body'].count('call     [Lokad.Onnx.ConvBlockedSpatial:MultiplyWinograd512('))
    assert sizes['current-captured-512']['multiply512_calls']==0 and sizes['candidate-captured-512']['multiply512_calls']==1
    value=dict(passed=True,mechanism_admitted=True,no_performance_measurement=True,first_capture_remains_failed=True,
        first_closure=pin(FIRST/'closed.json'),correction_closure=pin(FIX/'closed.json'),reviewer=pin(Path(__file__)),
        sizes=sizes,reductions=reviews,all_bodies=all_bodies,raw_files=raw_files,
        selected_body_counts={name:len(bodies) for name,bodies in selected.items()},
        first_resources=read(FIRST/'analysis.json')['resources'],correction_resources=read(FIX/'analysis.json')['resources'],
        limitations='The candidate needs eight separate broadcasts per paired reduction and a larger non-inlined helper. Both costs are retained. Diagnostic assembly proves the mechanism, not a complete-call or application gain. Damaged original bodies remain failed and are never reconstructed.')
    for row in raw_files:shutil.copy2(ROOT/row['source'],REPORT/row['file'])
    OUT.mkdir();(OUT/'review.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    (REPORT/'codegen-observations-20260923.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(review=pin(OUT/'review.json'),sizes=sizes,bodies=len(all_bodies),reviewed=len(reviews))))


if __name__=='__main__':main()
