"""Verify retained current-release/Parakeet proofs after the bounded cleanup."""
import hashlib,json,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/repository-retention-20260923'
PROOFS=['pyannote-winograd-product-root-amd-20260923/closed.json',
 'pyannote-winograd-product-app-amd-20260923/closed.json',
 'pyannote-winograd-product-shared-amd-v2-20260923/closed.json',
 'release-graph-baseline-amd-v2-20260923/closed.json',
 'parakeet-winograd-baseline-amd-20260923/closed.json',
 'parakeet-current-profile-amd-20260923/closed.json',
 'parakeet-short-dispatch-build-amd-v3-20260923/closed.json',
 'parakeet-short-dispatch-numerics-amd-v2-20260923/closed.json',
 'parakeet-short-dispatch-screen-amd-20260923/closed.json',
 'parakeet-dispatch-events-amd-20260923/closed.json',
 'parakeet-dispatch-full-export-amd-20260923/closed.json',
 'parakeet-wide-matmul-v3-20260921/capture-closed.json']
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def pin(p):
 with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def main():
 completion=read(BASE/'completed.json');assert completion['passed'] and not (BASE/'verified.json').exists()
 manifest=read(BASE/'proposed.json');assert pin(BASE/'proposed.json')['sha256']==completion['manifest_sha256']
 records={}
 for name in ['retired.jsonl','retired-python.jsonl']:
  for line in (BASE/name).read_text().splitlines():
   row=json.loads(line)
   assert row['manifest_sha256']==completion['manifest_sha256']
   if row['path'] in records:assert row==records[row['path']]
   records[row['path']]=row
 assert len(records)==len(manifest['files'])
 for row in manifest['files']:
  assert row['bytes']==records[row['path']]['bytes'] and row['mtime_ns']==records[row['path']]['mtime_ns']
  assert not (ROOT/row['path']).exists(),row['path']
 checked={};reports=[]
 for name in PROOFS:
  path=ROOT/'artifacts'/name;proof=read(path);assert proof['passed']
  # Older complete-capture/profile proofs use repository-relative paths,
  # including canonical models; newer campaign proofs use artifact-relative paths.
  scope=ROOT if any(n.startswith('artifacts/') for n in proof['files']) else path.parent
  for relative,wanted in proof['files'].items():
   file=scope/relative;key=str(file)
   if key not in checked:checked[key]=pin(file)
   assert checked[key]==wanted,(name,relative)
  reports.append(dict(file='artifacts/'+name,identity=pin(path),entries=len(proof['files'])))
 selected=read(ROOT/'artifacts/pyannote-winograd-product-root-amd-20260923/bundle/stage.json')
 inputs={n.removeprefix('source/'):w for n,w in selected['files'].items() if n.startswith('source/')};assert len(inputs)==420
 for name,wanted in inputs.items():assert pin(ROOT/name)==wanted,name
 totals={};count=0;reparse=[]
 scan_root=Path('\\\\?\\'+str(ROOT)) if os.name=='nt' else ROOT
 for current,dirs,names in os.walk(scan_root):
  ordinary=[]
  for name in dirs:
   directory=Path(current,name)
   if directory.is_symlink() or directory.is_junction():reparse.append(str(directory.relative_to(scan_root)))
   else:ordinary.append(name)
  dirs[:]=ordinary
  for name in names:
   p=Path(current,name);relative=p.relative_to(scan_root);n=p.stat().st_size;key=relative.parts[0]
   totals[key]=totals.get(key,0)+n;count+=1
 total=sum(totals.values());assert total<50_000_000_000,(total,totals)
 value=dict(passed=True,checked=time.time(),repo_bytes=total,repo_files=count,retired_files=len(records),retired_bytes=sum(r['bytes'] for r in records.values()),totals=totals,reparse_directories_excluded=reparse,closures=reports,unique_proof_files=len(checked),root_inputs=420,manifest=pin(BASE/'proposed.json'),journals={n:pin(BASE/n) for n in ['retired.jsonl','retired-python.jsonl']})
 (BASE/'verified.json').write_text(json.dumps(value,indent=2)+'\n')
 print(json.dumps({k:v for k,v in value.items() if k not in ['closures','totals','journals']},indent=2))
if __name__=='__main__':main()
