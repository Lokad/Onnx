"""Remove only closed VM transfer archives with byte-identical retained local inputs."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/parakeet-decoder-model-archive-headroom-20260927'
SURVEY = ROOT/'artifacts/parakeet-decoder-models-vm-archive-survey-20260927.json'


def pin(path):
    with path.open('rb') as f: return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(f, 'sha256').hexdigest())


def read(path): return json.loads(path.read_text())


def main():
    assert not OUT.exists(), 'One-time retirement; preserve any partial attempt'
    survey = read(SURVEY); assert survey['passed'] and len(survey['archives']) == 27
    wanted = {r['identity']['sha256'] for r in survey['archives']}; local = {}
    for folder in (ROOT/'artifacts').iterdir():
        path = folder/'payload.tar.gz'
        if not folder.is_dir() or not path.is_file() or not (folder/'prepared.json').is_file(): continue
        identity = pin(path)
        if identity['sha256'] not in wanted or read(folder/'prepared.json').get('archive') != identity: continue
        for name in ['closed.json', 'failed.json']:
            proof_path = folder/name
            if not proof_path.is_file(): continue
            proof = read(proof_path)
            if proof.get('files', {}).get('payload.tar.gz') != identity: continue
            if not (proof.get('passed') or proof.get('preserved_failure') or proof.get('terminal')): continue
            local[identity['sha256']] = dict(path=path.relative_to(ROOT).as_posix(), identity=identity,
                closure=proof_path.relative_to(ROOT).as_posix(), proof=pin(proof_path))
            break
    rows = [dict(row, retained=local[row['identity']['sha256']]) for row in survey['archives'] if row['identity']['sha256'] in local]
    assert len(rows) == 11 and sum(r['physical'] for r in rows) == 27578368
    script = REMOTE_SCRIPT.replace('@@ROWS@@', repr(rows))
    compile(script, 'decoder-model-staging-archive-retirement', 'exec')
    OUT.mkdir()
    def save(name, value):
        with (OUT/name).open('x') as f: json.dump(value, f, indent=2); f.write('\n')
    save('intent.json', dict(tool=pin(Path(__file__)), survey=pin(SURVEY), rows=rows))
    with (OUT/'remote.py').open('x', newline='\n') as f: f.write(script)
    loader = importlib.util.spec_from_file_location('archive_transport', ROOT/'tests/parakeet/ort-diagnosis-amd/run.py')
    transport = importlib.util.module_from_spec(loader); loader.loader.exec_module(transport)
    result = transport.ssh(script); save('result.json', result)
    assert result['passed']
    for row in rows:
        assert pin(ROOT/row['retained']['path']) == row['identity']
        assert pin(ROOT/row['retained']['closure']) == row['retained']['proof']
    save('closed.json', dict(passed=True, result=pin(OUT/'result.json'), intent=pin(OUT/'intent.json'),
        script=pin(OUT/'remote.py'), locally_retained=len(rows)))
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}))


REMOTE_SCRIPT = r'''
from pathlib import Path
import hashlib,json,os,signal,sys
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
os.sched_setaffinity(0,{0});signal.alarm(100)
def pin(path):
 with path.open('rb') as f:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def read(path):return json.loads(path.read_text())
def live(i):
 try:
  p=psutil.Process(i['pid']);return p.create_time()==i['birth'] and p.status()!=psutil.STATUS_ZOMBIE
 except psutil.NoSuchProcess:return False
own=psutil.Process();parents={own.pid,*[p.pid for p in own.parents()]}
assert psutil.boot_time()==1789634288.0
for p in psutil.process_iter(['name','cmdline']):
 if p.pid in parents:continue
 assert p.info['name'] not in ['dotnet','perf']
 assert not (p.info['name'].startswith('python') and '/dev/shm/lokad-' in ' '.join(p.info['cmdline'] or []))
raw=set();documents=0
for folder in Path('/dev/shm').glob('lokad-*'):
 if not folder.is_dir():continue
 for name in ['spec.json','payload.json','stage.json']:
  p=folder/name
  if not p.is_file():continue
  v=read(p);documents+=1;raw.update(str(folder/n) for n in v.get('files',{}));raw.update(v.get('external',{}))
  for link in v.get('links',{}).values():
   if isinstance(link,dict) and isinstance(link.get('source'),str):raw.add(link['source'])
protected={str(Path(n).resolve()) for n in raw};rows=@@ROWS@@
before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
for row in rows:
 p=Path(row['path']);base=p.parent
 assert p.resolve()==p and base.parent==Path('/dev/shm') and p.name=='transfer.tar.gz'
 assert not p.is_symlink() and str(p) not in protected and pin(p)==row['identity']
 s=p.stat();assert s.st_nlink==1 and s.st_blocks*512==row['physical']
 receipts=[read(base/n) for n in ['collection.json','build-collection.json','capture-collection.json'] if (base/n).is_file()]
 assert receipts and all(r.get('terminal') for r in receipts)
 assert all(not live(i) for r in receipts for i in r.get('identities',[]))
for row in rows:
 p=Path(row['path']);assert pin(p)==row['identity'];p.unlink()
print(json.dumps(dict(passed=True,files=rows,archives=len(rows),physical_bytes_freed=sum(r['physical'] for r in rows),
 protected_paths=len(protected),frozen_documents=documents,before=before,
 after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free))))
'''


if __name__ == '__main__': main()
