"""Retain every emitted body; census the four isolated first-use float kernel bodies."""
import difflib,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/parakeet-wide-entry-first-use-numerics-amd-20260923'
OUT=ROOT/'artifacts/parakeet-wide-entry-first-use-codegen-20260923'
METHODS={'PackPanelsB':'ShortWidePackPanelsB','mm_unsafe_vectorized_intrinsics_2x4packed_bump':'ShortWideMultiply2Rows','mm_unsafe_vectorized_intrinsics_3x4packed':'ShortWideMultiply3Rows','mm_unsafe_vectorized_intrinsics':'ShortWideMultiplyRemainder'}

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
    names = ['RunWideProjectionMatMul2DCore','RunIsolatedShortWidePackedRows','ShortWideMultiply2Rows','ShortWideMultiply3Rows']
    for name in names:
        candidate=[r for r in roles['candidate'] if ':'+name+'(' in r['method']]
        first_use[name]=dict(bodies=[r['index'] for r in candidate],tiers=[r['tier'] for r in candidate],
            passed=len(candidate)==1 and candidate[0]['tier']=='FullOpts' and '; optimized code' in candidate[0]['body'])
    # Packer and raw remainder may inline into the FullOpts helper.
    for name in ['ShortWidePackPanelsB','ShortWideMultiplyRemainder']:
        candidate=[r for r in roles['candidate'] if ':'+name+'(' in r['method']]
        first_use[name]=dict(bodies=[r['index'] for r in candidate],tiers=[r['tier'] for r in candidate],
            passed=all(r['tier']=='FullOpts' for r in candidate),standalone_optional=True)
    value=dict(closure=pin(BASE/'closed.json'),inputs=inputs,bodies=bodies,first_use=first_use,comparisons=comparisons,
        first_use_passed=all(r['passed'] for r in first_use.values()),manual_review_required=True,no_performance_measurement=True)
    (OUT/'census.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({k:v for k,v in value.items() if k!='bodies'}))
    print(json.dumps([{k:r[k] for k in ['role','index','method','tier','bytes']} for r in bodies]))

if __name__=='__main__':main()
