"""Review every complete-provider diagnostic body without attributing past clocks."""
import difflib
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'artifacts/parakeet-provider-where-fallback-codegen-amd-20260924'
OUT = ROOT / 'artifacts/parakeet-provider-where-fallback-codegen-review-20260924'
PARSER = ROOT / 'tests/parakeet/wide-entry-first-use-results/inspect_codegen.py'
spec = importlib.util.spec_from_file_location('parser', PARSER)
parser = importlib.util.module_from_spec(spec); spec.loader.exec_module(parser)
pin = parser.pin


def normalized(instructions):
    return [re.sub(r'0x[0-9A-Fa-f]{10,}', '<address>', i) for i in instructions]


def main():
    assert not OUT.exists()
    assert pin(BASE/'closed.json')['sha256'] == 'ae8b002e4d8fba223545ac1b535fa2f1a88cb503a2da744befa8735769c77ee4'
    proof = json.loads((BASE/'closed.json').read_text()); assert proof['passed']
    for name,wanted in proof['files'].items(): assert pin(BASE/name) == wanted,name
    OUT.mkdir(); bodies=[]; roles={}; inputs={}; entries={}; providers={}; callers={}
    for role,count in [('current',36),('candidate',35)]:
        path=BASE/'collected/logs'/(role+'-diagnostic-512.stdout')
        rows=parser.parse(path); assert len(rows)==count; roles[role]=rows; inputs[role]=pin(path)
        for row in rows:
            labels=re.findall(r'^(G_M\d+_IG\d+):',row['body'],re.M)
            assert len(labels)==len(set(labels)) and set(re.findall(r'G_M\d+_IG\d+',row['body'])) <= set(labels)
            target=OUT/f'{role}-{row["index"]}.txt'; target.write_text(row['body']+'\n')
            bodies.append(dict(role=role,**{k:row[k] for k in ['index','method','tier','bytes']},
                file=target.relative_to(ROOT).as_posix(),identity=pin(target),labels=len(labels),
                instructions=len(row['instructions']),calls=[i for i in row['instructions'] if i.startswith(('call ','tail.jmp'))]))
        full=[r for r in rows if '[float]:Where(' in r['method'] and r['tier']=='Tier1']; assert len(full)==1
        entry=full[0]; ins=entry['instructions']
        assert not any('UniformScalarWhere' in i or ':BroadcastShape(' in i for i in ins)
        assert sum(i.startswith(('idiv ','div ')) for i in ins)==4
        assert '; 23 inlinees with PGO data; 24 single block inlinees; 8 inlinees without PGO data' in entry['body']
        entries[role]=dict(index=entry['index'],bytes=entry['bytes'],external_shape_calls=0,uniform_calls=0,divisions=4,
            profile=[s for s in entry['body'].splitlines() if 'PGO' in s or 'inlinee' in s],
            tiers=[{k:r[k] for k in ['index','tier','bytes']} for r in rows if '[float]:Where(' in r['method']])
        provider=[r for r in rows if r['method'].startswith('Lokad.Onnx.CPUExecutionProvider:Where')]
        assert [r['tier'] for r in provider]==['Tier0','Instrumented Tier0','Tier1']
        whole=provider[-1]; ins=whole['instructions']
        for dtype in ['bool','byte','int','long','uint','ulong','float','double','System.Half']:
            assert sum(f'Lokad.Onnx.Tensor`1[{dtype}]:Where(' in i for i in ins)==1
        helper_calls=sum('UniformScalarWhere:Try' in i for i in ins)
        assert helper_calls==(1 if role=='candidate' else 0)
        if role=='candidate':
            ix=next(i for i,s in enumerate(ins) if 'UniformScalarWhere:Try' in s)
            assert any('0x1000' in s and s.startswith('cmp ') for s in ins[:ix])
            assert ins[ix+1]=='test     eax, eax' and ins[ix+2].startswith('jne ')
            assert sum(':Where(' in s for s in ins[ix+3:ix+15])==1
        providers[role]=dict(entries=[{k:r[k] for k in ['index','tier','bytes']} for r in provider],
            whole_bytes=whole['bytes'],helper_calls=helper_calls,all_nine_dtype_fallbacks=True,
            profile=[s for s in whole['body'].splitlines() if 'PGO' in s or 'inlinee' in s])
        consumer=[r for r in rows if r['method'].startswith('Program:Exercise[')]
        assert {r['method'].split('[')[1].split(']')[0] for r in consumer}=={'long','float','bool','byte','int','uint','ulong','double','System.Half'}
        for row in consumer:
            assert sum('CPUExecutionProvider:Where' in i for i in row['instructions'])==1
            assert not any('UniformScalarWhere' in i for i in row['instructions'])
        callers[role]=dict(bodies=len(consumer),all_nine_dtypes=True,complete_provider_called=True,helper_not_called_directly=True)
    helper=next(r for r in roles['candidate'] if 'UniformScalarWhere:Try' in r['method'])
    old=ROOT/'artifacts/parakeet-scalar-where-numerics-amd-v3-20260924/collected/logs/candidate-codegen-512.stdout'
    old_helper=next(r for r in parser.parse(old) if 'UniformScalarWhere:Try' in r['method'])
    assert helper['tier']=='FullOpts' and helper['bytes']==1277
    assert normalized(helper['instructions'])==normalized(old_helper['instructions'])
    for token in ['SpanHelpers+Negate`1[byte]','SpanHelpers+DontNegate`1[byte]','SpanHelpers:Fill[float]','SpanHelpers:Memmove']:
        assert sum(token in i for i in helper['instructions'])==1
    full={role:next(r for r in rows if '[float]:Where(' in r['method'] and r['tier']=='Tier1') for role,rows in roles.items()}
    diff=list(difflib.unified_diff(normalized(full['current']['instructions']),normalized(full['candidate']['instructions']),n=3))
    result=dict(passed=True,no_performance_measurement=True,prior_screen_remains_rejected=True,historical_timed_tiers_proven=False,
        closure=pin(BASE/'closed.json'),inputs=inputs,parser=pin(PARSER),reviewer=pin(Path(__file__)),bodies=bodies,
        whole_float_entries=entries,providers=providers,consumers=callers,helper_normalized_exact_to_v3=True,
        helper_reference=pin(old),generic_native_diff=diff,
        conclusion='Both providers emit complete Tier1 bodies under this call history. Both generic float entries retain 23 profiled inlinees, no external BroadcastShape calls and four divisions. Native layouts differ. Candidate still scans mixed masks and then calls the generic per-element broadcast fallback. This rules out absence of a candidate provider Tier1 body in this diagnostic, not in the historical timed workers.',
        next_work='Design a distinct dense scalar selection helper that processes mixed masks in contiguous broadcast runs, preserving uniform copy/fill and all bit/ownership semantics. Retain the rejected screen; any subsequent measurement redesign must first qualify selected/selected stability.')
    target=Path(__file__).parent/'fallback-codegen-review-20260924.json'
    assert not target.exists(); target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(review=pin(target),bodies=len(bodies),entries=entries,providers=providers,consumers=callers)))


if __name__=='__main__': main()
