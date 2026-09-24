"""Review all untimed fallback bodies; do not attribute historical timed tiers."""
import importlib.util
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/parakeet-scalar-where-fallback-codegen-amd-20260924'
OUT=ROOT/'artifacts/parakeet-scalar-where-fallback-codegen-review-20260924'
PARSER=ROOT/'tests/parakeet/wide-entry-first-use-results/inspect_codegen.py'
spec=importlib.util.spec_from_file_location('retained_parser',PARSER)
parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
pin=parser.pin


def main():
    assert not OUT.exists()
    assert pin(BASE/'closed.json')['sha256']=='1cf6a1241cb1501fdf24e7dc049b19181353f3d80c9dfacd71729ece384ffd41'
    proof=json.loads((BASE/'closed.json').read_text());assert proof['passed']
    for name,wanted in proof['files'].items():assert pin(BASE/name)==wanted,name
    OUT.mkdir();roles={};bodies=[];entries={};callers={};inputs={}
    for role,count in [('current',33),('candidate',32)]:
        path=BASE/'collected/logs'/(role+'-diagnostic-512.stdout');inputs[role]=pin(path)
        rows=parser.parse(path);assert len(rows)==count;roles[role]=rows
        for row in rows:
            labels=re.findall(r'^(G_M\d+_IG\d+):',row['body'],re.M)
            assert len(labels)==len(set(labels)) and set(re.findall(r'G_M\d+_IG\d+',row['body']))<=set(labels)
            target=OUT/f'{role}-{row["index"]}.txt';target.write_text(row['body']+'\n',encoding='utf8')
            bodies.append(dict(role=role,**{k:row[k] for k in ['index','method','tier','bytes']},
                file=target.relative_to(ROOT).as_posix(),identity=pin(target),labels=len(labels),instructions=len(row['instructions']),
                calls=[i for i in row['instructions'] if i.startswith(('call ','tail.jmp'))]))
        full=[r for r in rows if '[float]:Where(' in r['method'] and r['tier']=='Tier1'];assert len(full)==1
        entry=full[0];ins=entry['instructions']
        entries[role]=dict(index=entry['index'],bytes=entry['bytes'],
            shape_calls=[i for i in ins if i.startswith('call ') and ':BroadcastShape(' in i],
            uniform_calls=[i for i in ins if 'UniformScalarWhere:Try' in i],
            divisions=sum(i.startswith(('idiv ','div ')) for i in ins),
            profile=[line for line in entry['body'].splitlines() if 'PGO' in line or 'inlinee' in line],
            tiers=[dict(index=r['index'],tier=r['tier'],bytes=r['bytes']) for r in rows if '[float]:Where(' in r['method']])
        caller=[r for r in rows if r['method'].startswith('Program:Exercise[')]
        assert {r['method'].split('[')[1].split(']')[0] for r in caller}=={'long','float','bool','byte','int','uint','ulong','double','System.Half'}
        for row in caller:
            assert sum(i.startswith('call ') and ':Where(' in i for i in row['instructions'])==1
            assert not any('UniformScalarWhere' in i for i in row['instructions'])
        callers[role]=dict(bodies=len(caller),all_nine_dtypes=True,direct_public_where_calls=True,helper_not_called_directly=True)
    assert entries['current']['bytes']==2893 and entries['candidate']['bytes']==2426
    assert len(entries['current']['shape_calls'])==0 and len(entries['candidate']['shape_calls'])==3
    assert len(entries['current']['uniform_calls'])==0 and len(entries['candidate']['uniform_calls'])==1
    assert entries['current']['divisions']==entries['candidate']['divisions']==4
    helper=next(r for r in roles['candidate'] if r['method'].startswith('Lokad.Onnx.UniformScalarWhere:Try[float]'))
    assert helper['tier']=='FullOpts' and helper['bytes']==1277
    for token in ['SpanHelpers+Negate`1[byte]','SpanHelpers+DontNegate`1[byte]','SpanHelpers:Fill[float]','SpanHelpers:Memmove']:
        assert sum(token in i for i in helper['instructions'])==1
    assert not any(i.startswith(('idiv ','div ','vadd','vmul','vfmadd','vsub','vdiv')) for i in helper['instructions'])
    result=dict(passed=True,no_performance_measurement=True,prior_screen_remains_rejected=True,
        historical_timed_tiers_proven=False,closure=pin(BASE/'closed.json'),parser=pin(PARSER),reviewer=pin(Path(__file__)),
        inputs=inputs,bodies=bodies,whole_float_entries=entries,consumers=callers,
        conclusion='Both diagnostics emit a whole-method float Tier1 entry as well as OSR bodies. Selected Tier1 has no out-of-line BroadcastShape calls; candidate Tier1 has three, with six fewer profiled inlinees. The caller always invokes public Where. The generic insertion demonstrably changes compiled fallback structure in this diagnostic. It does not prove which tiers ran in the rejected screen or explain its full regression magnitude.',
        next_hypothesis='Move the uniform specialization to the existing CPU execution-provider float Where branch, preserving the entire public generic Tensor.Where method. Qualify a distinct source and both public boundaries; do not change the rejected screen or reuse its target gains as admission.')
    target=Path(__file__).parent/'fallback-codegen-review-20260924.json';assert not target.exists()
    target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(review=pin(target),bodies=len(bodies),whole_entries=entries,consumers=callers)))


if __name__=='__main__':main()
