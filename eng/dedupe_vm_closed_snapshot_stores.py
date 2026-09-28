"""Consolidate identical files in two completed VM snapshot stores, preserving paths."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.util.spec_from_file_location('snapshot_transport', ROOT/'eng/offload_vm_snapshots.py')
engine = importlib.util.module_from_spec(loader); loader.loader.exec_module(engine)
OUT = ROOT/'artifacts/vm-closed-snapshot-dedup-20260928'
STORES = [('vm-snapshot-offload-20260927', 'tmpfs-snapshots-20260927'),
          ('vm-pointwise-closed-snapshot-offload-20260927', 'pointwise-closed-snapshots-20260927')]


def script():
    proofs = {}
    for local, remote in STORES:
        base = ROOT/'artifacts'/local
        completed = json.loads((base/'completed.json').read_text())
        plan = json.loads((base/'plan.json').read_text())
        assert completed['passed'] and completed['files'] == len(plan['files'])
        proofs['/home/vermorel/Onnx/artifacts/'+remote] = dict(completed=completed, files=plan['files'])
    prelude = engine.SCRIPT.split('store=Path(', 1)[0]
    return prelude + '\nPROOFS=' + repr(proofs) + r'''
def pin(p):
 with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
rows=[]
for name,proof in PROOFS.items():
 store=Path(name);assert store.resolve()==store and store.parent==Path('/home/vermorel/Onnx/artifacts')
 assert json.loads((store/'completed.json').read_text())==proof['completed']
 for row in proof['files']:
  p=Path(row['target']);original=Path(row['path'])
  assert p.is_relative_to(store) and not p.is_symlink() and p.resolve()==p
  assert original.is_symlink() and original.resolve()==p and pin(p)==row['identity']
  if p.stat().st_size>=65536:rows.append(row)
if ACTION=='plan':
 groups={}
 for row in rows:groups.setdefault((row['identity']['bytes'],row['identity']['sha256']),[]).append(row)
 changes=[]
 for group in groups.values():
  if len(group)<2:continue
  source=Path(group[0]['target']);ss=source.stat()
  for row in group[1:]:
   target=Path(row['target']);ts=target.stat()
   assert ss.st_dev==ts.st_dev
   if ss.st_ino==ts.st_ino or ts.st_nlink!=1:continue
   changes.append(dict(source=str(source),target=str(target),identity=row['identity'],
    device=ts.st_dev,inode=ts.st_ino,allocated=ts.st_blocks*512))
 print(json.dumps(dict(passed=True,changes=changes,allocated=sum(r['allocated'] for r in changes),
  disk_free=psutil.disk_usage('/home/vermorel/Onnx').free)))
elif ACTION=='apply':
 allowed={r['target']:r['identity'] for r in rows}
 for row in PLAN['changes']:
  source=Path(row['source']);target=Path(row['target']);st=target.stat()
  assert allowed[str(source)]==allowed[str(target)]==row['identity']==pin(source)==pin(target)
  assert st.st_nlink==1 and (st.st_dev,st.st_ino,st.st_blocks*512)==(row['device'],row['inode'],row['allocated'])
 before=psutil.disk_usage('/home/vermorel/Onnx').free
 journal=Path('/home/vermorel/Onnx/artifacts/closed-snapshot-dedup-20260928.jsonl')
 with journal.open('x') as stream:
  for row in PLAN['changes']:
   source=Path(row['source']);target=Path(row['target']);link=target.with_name(target.name+'.dedup-link')
   assert not link.exists() and not link.is_symlink() and pin(target)==pin(source)==row['identity']
   os.link(source,link);os.replace(link,target)
   assert pin(target)==row['identity'] and target.stat().st_ino==source.stat().st_ino
   stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
 for row in rows:
  assert pin(Path(row['target']))==row['identity'] and Path(row['path']).resolve()==Path(row['target'])
 print(json.dumps(dict(passed=True,files=len(PLAN['changes']),allocated_reclaimed=PLAN['allocated'],
  disk_before=before,disk_after=psutil.disk_usage('/home/vermorel/Onnx').free,
  original_paths_and_bytes_preserved=True,journal=pin(journal))))
'''


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ['plan', 'apply']
    code = script(); identity = hashlib.sha256(code.encode()).hexdigest()
    if sys.argv[1] == 'plan':
        assert not OUT.exists()
        result = json.loads(engine.run.ssh('ACTION="plan"\n'+code, 180))
        OUT.mkdir(); engine.save(OUT/'plan.json', dict(script_sha256=identity, **result))
        print(json.dumps({k:v for k,v in result.items() if k != 'changes'}))
    else:
        assert not (OUT/'started.json').exists()
        plan = json.loads((OUT/'plan.json').read_text()); assert plan['script_sha256'] == identity
        engine.save(OUT/'started.json', dict(script_sha256=identity))
        result = json.loads(engine.run.ssh('ACTION="apply"\nPLAN='+repr(plan)+'\n'+code, 300))
        engine.save(OUT/'completed.json', result); print(json.dumps(result))
