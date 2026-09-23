"""Retain every emitted body; census the four first-use float kernel bodies."""
import difflib,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/parakeet-first-use-kernels-numerics-amd-20260923'
OUT=ROOT/'artifacts/parakeet-first-use-kernels-codegen-20260923'
METHODS=['PackPanelsB','mm_unsafe_vectorized_intrinsics_2x4packed_bump','mm_unsafe_vectorized_intrinsics_3x4packed','mm_unsafe_vectorized_intrinsics']

def pin(p):
    with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())

def parse(path):
    text=path.read_text()
    matches=list(re.finditer(r'^; Assembly listing for method ([^\n]+) \(([^\n]+)\)\n(.*?); Total bytes of code (\d+)',text,re.M|re.S))
    assert len(matches)==text.count('; Assembly listing for method ')==text.count('; Total bytes of code ')
    return [dict(index=i,method=m[1],tier=m[2],bytes=int(m[4]),body=m[0],
        instructions=[line.strip() for line in m[3].splitlines() if re.match(r'^       [a-z]',line)]) for i,m in enumerate(matches)]

def main():
    assert not OUT.exists()
    closure=json.loads((BASE/'closed.json').read_text());assert closure['passed'] and closure['numerically_admitted']
    for name,wanted in closure['files'].items():assert pin(BASE/name)==wanted,name
    OUT.mkdir();roles={};bodies=[];inputs={}
    for role in ['current','candidate']:
        path=BASE/'collected/logs'/(role+'-codegen-512.stdout');inputs[role]=pin(path)
        roles[role]=parse(path)
        for row in roles[role]:
            (OUT/(role+'-'+str(row['index'])+'.txt')).write_text(row['body']+'\n')
            bodies.append(dict(role=role,**{k:row[k] for k in ['index','method','tier','bytes']},
                instruction_count=len(row['instructions']),calls=[s for s in row['instructions'] if s.startswith(('call ','tail.jmp'))],
                vector_fma=sum('vfmadd' in s for s in row['instructions'])))
    first_use={};comparisons=[]
    for name in METHODS:
        select=lambda role:[r for r in roles[role] if r['method'].startswith('Lokad.Onnx.MathOps:'+name+'(')]
        candidate=select('candidate');current=select('current')
        first_use[name]=dict(bodies=[r['index'] for r in candidate],tiers=[r['tier'] for r in candidate],
            passed=len(candidate)==1 and candidate[0]['tier']=='FullOpts' and '; optimized code' in candidate[0]['body'])
        left=[r for r in current if r['tier']=='Tier1']
        if len(left)==len(candidate)==1:
            diff=list(difflib.unified_diff(left[0]['body'].splitlines(),candidate[0]['body'].splitlines(),fromfile='current-Tier1',tofile='candidate-FullOpts',lineterm=''))
            (OUT/(name+'.diff')).write_text('\n'.join(diff)+'\n')
            comparisons.append(dict(method=name,current_index=left[0]['index'],candidate_index=candidate[0]['index'],
                current_bytes=left[0]['bytes'],candidate_bytes=candidate[0]['bytes'],identical_instructions=left[0]['instructions']==candidate[0]['instructions']))
    value=dict(closure=pin(BASE/'closed.json'),inputs=inputs,bodies=bodies,first_use=first_use,comparisons=comparisons,
        first_use_passed=all(r['passed'] for r in first_use.values()),manual_review_required=True,no_performance_measurement=True)
    (OUT/'census.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({k:v for k,v in value.items() if k!='bodies'}))
    print(json.dumps([{k:r[k] for k in ['role','index','method','tier','bytes']} for r in bodies]))

if __name__=='__main__':main()
