"""Extract every emitted body and exact consumer instruction differences."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/parakeet-short-dispatch-numerics-amd-20260923'
OUT=ROOT/'artifacts/parakeet-short-dispatch-codegen-review-20260923'
def pin(p):
 with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def parse(path):
 text=path.read_text()
 matches=list(re.finditer(r'^; Assembly listing for method ([^\n]+) \(([^\n]+)\)\n(.*?); Total bytes of code (\d+)',text,re.M|re.S))
 assert len(matches)==text.count('; Assembly listing for method ')==text.count('; Total bytes of code ')
 rows=[]
 for index,m in enumerate(matches):
  # Restrict the header to its first line despite the DOTALL flag.
  assert '\n' not in m[1]
  instructions=[line.strip() for line in m[3].splitlines() if re.match(r'^       [a-z]',line)]
  rows.append(dict(index=index,method=m[1],tier=m[2],bytes=int(m[4]),instructions=instructions,body=m[0]))
 return rows
assert not OUT.exists();OUT.mkdir()
inputs={};roles={};bodies=[]
for role in ['current','candidate']:
 path=BASE/'collected/logs'/(role+'-codegen-512.stdout');inputs[role]=pin(path)
 rows=parse(path);roles[role]=rows
 for r in rows:
  (OUT/(role+'-'+str(r['index'])+'.txt')).write_text(r['body'])
  bodies.append(dict(role=role,**{k:r[k] for k in ['index','method','tier','bytes']},
    instruction_count=len(r['instructions']),calls=[s for s in r['instructions'] if s.startswith(('call ','tail.jmp'))],
    vector_fma=sum('vfmadd' in s for s in r['instructions'])))
diffs=[]
for ending in ['2x4packed_bump','3x4packed']:
 left=[r for r in roles['current'] if ending+'(' in r['method'] and r['tier']=='Tier1'];right=[r for r in roles['candidate'] if ending+'(' in r['method'] and r['tier']=='Tier1']
 assert len(left)==len(right)==1
 a=left[0]['instructions'];b=right[0]['instructions'];assert len(a)==len(b)
 differences=[dict(index=i,current=x,candidate=y,next=a[i+1]) for i,(x,y) in enumerate(zip(a,b)) if x!=y]
 diffs.append(dict(method=ending,current_bytes=left[0]['bytes'],candidate_bytes=right[0]['bytes'],differences=differences))
value=dict(inputs=inputs,bodies=bodies,consumer_differences=diffs)
(OUT/'census.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(dict(census=str(OUT/'census.json'),bodies=[{k:r[k] for k in ['role','index','method','tier','bytes']} for r in bodies],consumer_differences=diffs)))
