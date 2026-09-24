"""Retain complete bodies and enforce the prospective whole-method entry gate."""
import difflib
import importlib.util
import json
import re
from pathlib import Path
from protocol import ORDER,pin,read,save
from prepare import ROOT,BASE

OUT=ROOT/'artifacts/parakeet-dense-scalar-where-balanced-native-20260924'
REPORT=ROOT/'tests/parakeet/dense-scalar-where-results/balanced-codegen-review-20260924.json'
PARSER=ROOT/'tests/parakeet/wide-entry-first-use-results/inspect_codegen.py'
spec=importlib.util.spec_from_file_location('native_parser',PARSER)
parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
DTYPES=['bool','byte','int','long','uint','ulong','float','double','System.Half']


def main():
    assert not OUT.exists() and not REPORT.exists()
    closure=read(BASE/'closed.json');assert closure['passed'] and not closure['performance_admitted']
    for name,wanted in closure['files'].items():assert pin(BASE/name)==wanted,name
    OUT.mkdir();bodies=[];processes={};inputs={};gates=[];comparisons=[]
    for process in ORDER:
        path=BASE/'collected/logs'/(process+'.stdout');inputs[process]=pin(path);rows=parser.parse(path);assert rows
        processes[process]=rows
        for row in rows:
            labels=re.findall(r'^(G_M\d+_IG\d+):',row['body'],re.M)
            assert len(labels)==len(set(labels)) and set(re.findall(r'G_M\d+_IG\d+',row['body']))<=set(labels)
            target=OUT/f'{process}-{row["index"]}.txt';target.write_text(row['body']+'\n');ins=row['instructions']
            bodies.append(dict(process=process,**{k:row[k] for k in ['index','method','tier','bytes']},
                file=target.relative_to(ROOT).as_posix(),identity=pin(target),labels=len(labels),instructions=len(ins),
                calls=[s for s in ins if s.startswith(('call ','tail.jmp'))],
                divisions=sum(s.startswith(('idiv ','div ')) for s in ins),
                range_checks=sum('RNGCHKFAIL' in s for s in ins),
                profile=[s for s in row['body'].splitlines() if any(t in s for t in ['PGO','inlinee','OSR','optimized','minopt'])]))
        names=['Program:MeasureBatch(','Lokad.Onnx.CPUExecutionProvider:Where(']+[f'Lokad.Onnx.Tensor`1[{dtype}]:Where(' for dtype in DTYPES]
        for name in names:
            entries=[r for r in rows if r['method'].startswith(name) and r['tier']=='Tier1']
            passed=len(entries)==1
            if name=='Program:MeasureBatch(' and passed:
                ins=entries[0]['instructions']
                passed=sum('CPUExecutionProvider:Where(' in s for s in ins)==1 and sum('GetTimestamp()' in s and s.startswith('call ') for s in ins)==2
                passed=passed and not any('DenseScalarWhere' in s for s in ins)
            gates.append(dict(process=process,method=name,passed=passed,entries=[{k:r[k] for k in ['index','tier','bytes']} for r in entries]))
    left={r['method']:r for r in processes[ORDER[0]] if r['tier']=='Tier1'}
    right={r['method']:r for r in processes[ORDER[1]] if r['tier']=='Tier1'}
    for i,method in enumerate(sorted(set(left)|set(right))):
        a=left.get(method);b=right.get(method)
        row=dict(method=method,current=None if a is None else a['bytes'],second=None if b is None else b['bytes'])
        if a is not None and b is not None:
            normalize=lambda lines:[re.sub(r'\(reloc 0x[0-9a-fA-F]+\)','(reloc <address>)',s) for s in lines]
            delta=list(difflib.unified_diff(normalize(a['instructions']),normalize(b['instructions']),fromfile=ORDER[0],tofile=ORDER[1],lineterm=''))
            path=OUT/f'diff-{i}.txt';path.write_text('\n'.join(delta)+'\n')
            row.update(file=path.relative_to(ROOT).as_posix(),identity=pin(path))
        comparisons.append(row)
    result=dict(passed=True,native_admitted=all(g['passed'] for g in gates),diagnostic_only=True,performance_admitted=False,
        closure=pin(BASE/'closed.json'),inputs=inputs,parser=pin(PARSER),reviewer=pin(Path(__file__)),bodies=bodies,gates=gates,comparisons=comparisons,
        normalization='Only explicit relocation fields; all other constants, registers, instructions, branch targets and call identities retained.',
        limitation='Diagnostic entries do not prove tiers in an untraced control. Native admission only permits that separately frozen control.')
    save(REPORT,result)
    print(json.dumps(dict(review=pin(REPORT),bodies=len(bodies),native_admitted=result['native_admitted'],gates=gates)))


if __name__=='__main__':main()
