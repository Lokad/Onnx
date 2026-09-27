"""Move inactive tmpfs snapshots to disk, preserving every byte and original path."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/vm-snapshot-offload-20260927'
sys.path.insert(0, str(ROOT / 'tests/parakeet/decoder-lstm-layout-root-amd'))
import run

SCRIPT = r'''
from pathlib import Path
import hashlib,json,os,shutil,sys
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
os.sched_setaffinity(0,{0})
assert psutil.boot_time()==1789634288.0
me=psutil.Process();ancestors={me.pid,*[p.pid for p in me.parents()]}
for p in psutil.process_iter(['pid','name','cmdline']):
 if p.pid in ancestors:continue
 assert p.info['name'] not in ['dotnet','perf'],p.info
 assert not (p.info['name'].startswith('python') and '/dev/shm/lokad-' in ' '.join(p.info['cmdline'] or [])),p.info
store=Path('/home/vermorel/Onnx/artifacts/tmpfs-snapshots-20260927')

def pin(p):
 with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())

def eligible(p):
 if p.is_symlink() or not p.is_file() or p.resolve()!=p:return False
 rel=p.relative_to('/dev/shm');parts=rel.parts
 if len(parts)<2 or not parts[0].startswith('lokad-') or parts[0].startswith('lokad-lstmlayout'):return False
 if p.stat().st_nlink!=1 or p.stat().st_size<4096:return False
 # Only inactive snapshots and metadata. Do not relocate installed runtimes,
 # canonical models, current release evidence, audio assets or the offline feed.
 return (len(parts)==2 and parts[1] in {'payload.json','stage.json','spec.json','collection.json'}
  or parts[1] in {'source','evidence'}
  or parts[0]=='lokad-pyannote-blocked-spatial-product-20260922' and parts[1]=='fixtures')

def resources():
 return dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free,disk=psutil.disk_usage('/home/vermorel/Onnx').free)

if ACTION=='plan':
 assert not store.exists()
 rows=[]
 for folder in Path('/dev/shm').glob('lokad-*'):
  for p in folder.rglob('*'):
   if not eligible(p):continue
   st=p.stat();rows.append(dict(path=str(p),target=str(store/p.relative_to('/dev/shm')),
    identity=pin(p),device=st.st_dev,inode=st.st_ino,allocated=st.st_blocks*512))
 assert rows
 result=dict(files=rows,resources=resources(),allocated=sum(r['allocated'] for r in rows),
  policy='Relocate single-link inactive snapshots to disk with atomic symlinks; all original paths and bytes remain readable.')
 print(json.dumps(result))
elif ACTION=='apply':
 assert not store.exists()
 assert resources()['disk']>PLAN['allocated']+2*1024**3
 for row in PLAN['files']:
  p=Path(row['path']);assert eligible(p) and pin(p)==row['identity']
  st=p.stat();assert (st.st_dev,st.st_ino,st.st_blocks*512)==(row['device'],row['inode'],row['allocated'])
  target=Path(row['target']);assert target==store/p.relative_to('/dev/shm') and not target.exists()
 store.mkdir()
 assert store.stat().st_dev!=Path('/dev/shm').stat().st_dev
 (store/'plan.json').write_text(json.dumps(PLAN))
 before=resources()
 with (store/'journal.jsonl').open('x') as journal:
  for row in PLAN['files']:
   p=Path(row['path']);target=Path(row['target']);target.parent.mkdir(parents=True,exist_ok=True)
   with p.open('rb') as source,target.open('xb') as dest:
    shutil.copyfileobj(source,dest);dest.flush();os.fsync(dest.fileno())
   shutil.copystat(p,target)
   assert pin(target)==row['identity'] and pin(p)==row['identity']
   link=p.with_name(p.name+'.offload-link');assert not link.exists() and not link.is_symlink()
   link.symlink_to(target);os.replace(link,p)
   assert p.is_symlink() and p.resolve()==target and pin(p)==row['identity']
   journal.write(json.dumps(row)+'\n');journal.flush();os.fsync(journal.fileno())
 for row in PLAN['files']:
  p=Path(row['path']);assert p.is_symlink() and p.resolve()==Path(row['target']) and pin(p)==row['identity']
 result=dict(passed=True,files=len(PLAN['files']),allocated_moved=PLAN['allocated'],before=before,after=resources(),
  preservation='All bytes retained on disk, readable through their original paths; no proof or model content deleted.')
 (store/'completed.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result))
'''


def save(path, value):
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2)


def main(action):
    identity = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if action == 'plan':
        assert not OUT.exists()
        value = json.loads(run.ssh('ACTION="plan"\n' + SCRIPT, 180))
        OUT.mkdir()
        save(OUT / 'plan.json', dict(driver=identity, **value))
        print(json.dumps(dict(files=len(value['files']), allocated=value['allocated'], resources=value['resources'])))
    else:
        assert not (OUT / 'started.json').exists()
        plan = json.loads((OUT / 'plan.json').read_text())
        assert plan['driver'] == identity
        save(OUT / 'started.json', dict(driver=identity))
        value = json.loads(run.ssh('ACTION="apply"\nPLAN=' + repr(plan) + '\n' + SCRIPT, 300))
        save(OUT / 'completed.json', value)
        print(json.dumps(value))


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ['plan', 'apply']
    main(sys.argv[1])
