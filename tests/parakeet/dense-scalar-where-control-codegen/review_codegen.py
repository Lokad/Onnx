"""Retain every complete diagnostic body and compare strict instruction listings."""
import collections
import difflib
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from protocol import ORDER,pin,read,save
from prepare import ROOT,BASE,CONSUMER

OUT=ROOT/'artifacts/parakeet-dense-scalar-where-control-native-20260924'
REPORT=ROOT/'tests/parakeet/dense-scalar-where-results/control-codegen-review-20260924.json'
PARSER=ROOT/'tests/parakeet/wide-entry-first-use-results/inspect_codegen.py'
spec=importlib.util.spec_from_file_location('native_parser',PARSER)
parser=importlib.util.module_from_spec(spec); spec.loader.exec_module(parser)


def normalized(instructions):
    # Only fields explicitly identified as relocations by the runtime are erased.
    # Unmarked pointer-looking immediates remain: they can also be real constants.
    return [re.sub(r'\(reloc 0x[0-9a-fA-F]+\)', '(reloc <address>)', s) for s in instructions]


def main():
    assert not OUT.exists() and not REPORT.exists()
    proof=read(BASE/'closed.json'); assert proof['passed'] and not proof['performance_admitted']
    for name,wanted in proof['files'].items(): assert pin(BASE/name)==wanted,name
    analysis=read(BASE/'analysis.json'); assert analysis['consumer']==CONSUMER and analysis['diagnostic_only']
    OUT.mkdir(); bodies=[]; processes={}; inputs={}; comparisons=[]; important={}
    for process in ORDER:
        path=BASE/'collected/logs'/(process+'.stdout'); rows=parser.parse(path); assert rows
        inputs[process]=pin(path); processes[process]=rows; important[process]={}
        seen=collections.Counter()
        for row in rows:
            labels=re.findall(r'^(G_M\d+_IG\d+):',row['body'],re.M)
            assert len(labels)==len(set(labels)) and set(re.findall(r'G_M\d+_IG\d+',row['body']))<=set(labels)
            ins=row['instructions']; norm=normalized(ins)
            key=(row['method'],row['tier']); occurrence=seen[key]; seen[key]+=1
            row['key']=(*key,occurrence)
            target=OUT/f'{process}-{row["index"]}.txt'; target.write_text(row['body']+'\n')
            bodies.append(dict(process=process,**{k:row[k] for k in ['index','method','tier','bytes']},occurrence=occurrence,
                file=target.relative_to(ROOT).as_posix(),identity=pin(target),labels=len(labels),
                instructions=len(ins),instruction_sha256=hashlib.sha256('\n'.join(ins).encode()).hexdigest(),
                relocation_normalized_sha256=hashlib.sha256('\n'.join(norm).encode()).hexdigest(),
                calls=[s for s in ins if s.startswith(('call ','tail.jmp'))],
                divisions=sum(s.startswith(('idiv ','div ')) for s in ins),
                range_check_calls=sum('RNGCHKFAIL' in s for s in ins),
                type_check_calls=[s for s in ins if any(t in s for t in ['CHKCAST','ISINSTANCE','CLASSPROFILE'])],
                profile=[s for s in row['body'].splitlines() if any(t in s for t in ['PGO','inlinee','OSR','optimized','minopt'])],
                whole_tier1=row['tier']=='Tier1',osr='OSR' in row['tier']))
        for label,match in [
            ('provider',lambda r:r['method'].startswith('Lokad.Onnx.CPUExecutionProvider:Where(')),
            ('float_where',lambda r:r['method'].startswith('Lokad.Onnx.Tensor`1[float]:Where(')),
            ('float_phase',lambda r:'Work`1[float]:Phase(' in r['method'])]:
            chosen=[r for r in rows if match(r)]; assert chosen,(process,label)
            important[process][label]=[{k:r[k] for k in ['index','method','tier','bytes']} for r in chosen]
    first={r['key']:r for r in processes[ORDER[0]]}
    for process in ORDER[1:]:
        other={r['key']:r for r in processes[process]}
        for index,key in enumerate(sorted(set(first)|set(other))):
            a=first.get(key); b=other.get(key)
            record=dict(process=process,method=key[0],tier=key[1],occurrence=key[2],
                first_index=None if a is None else a['index'],other_index=None if b is None else b['index'],
                first_bytes=None if a is None else a['bytes'],other_bytes=None if b is None else b['bytes'])
            if a is not None and b is not None:
                left=normalized(a['instructions']); right=normalized(b['instructions'])
                delta=list(difflib.unified_diff(left,right,fromfile=ORDER[0],tofile=process,lineterm=''))
                target=OUT/f'diff-{process}-{index}.txt'; target.write_text('\n'.join(delta)+'\n')
                record.update(exact_instructions=a['instructions']==b['instructions'],explicit_relocations_only_equal=left==right,
                    diff=target.relative_to(ROOT).as_posix(),diff_identity=pin(target))
            comparisons.append(record)
    save(REPORT,dict(passed=True,diagnostic_only=True,performance_admitted=False,manual_review_required=True,
        closure=pin(BASE/'closed.json'),consumer=CONSUMER,inputs=inputs,parser=pin(PARSER),reviewer=pin(Path(__file__)),
        bodies=bodies,important=important,comparisons=comparisons,
        normalization='Only explicit (reloc 0x...) fields. All other immediates, registers, opcodes, branches and call identities retained. Diffs may include unmarked relocated addresses; they do not by themselves prove changed logic.',
        limitation='These bodies accompany traced fresh processes. They do not establish historical untraced timed tiers or causes. Tier1-OSR is distinct from whole-method Tier1.'))
    print(json.dumps(dict(review=pin(REPORT),bodies=len(bodies),important=important)))


if __name__=='__main__': main()
