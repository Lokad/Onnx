"""Retire closed VM tensor duplicates; keep every exact local correctness output."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT/'artifacts/parakeet-decoder-packed-row-models-amd-20260927'
OUT = ROOT/'artifacts/parakeet-decoder-shared-headroom-20260927'


def read(p): return json.loads(p.read_text())


def pin(p):
    with p.open('rb') as f: return dict(bytes=p.stat().st_size, sha256=hashlib.file_digest(f, 'sha256').hexdigest())


def main():
    assert not OUT.exists(), 'One-time retirement; preserve any partial attempt'
    assert pin(MODEL/'closed.json')['sha256'] == '5d6832083d103bef9db7bb733b1e96decbae22625f3138680ab6f9a426c78e3b'
    closed = read(MODEL/'closed.json'); receipt_path = MODEL/'collected/collection.json'
    assert closed['passed'] and closed['files']['collected/collection.json'] == pin(receipt_path)
    receipt = read(receipt_path); assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
    jobs = {f'{role}-native-{isa}' for role in ['selected', 'candidate'] for isa in ['512', '256']}
    rows = []
    for name, wanted in receipt['files'].items():
        parts = Path(name).parts
        if len(parts) != 3 or parts[0] not in jobs or parts[1] != 'result.json.tensors': continue
        assert pin(MODEL/'collected'/name) == wanted == closed['files']['collected/'+name]
        rows.append(dict(name=name, identity=wanted))
    assert len(rows) == 3136
    script = REMOTE.replace('@@ROWS@@', repr(rows)).replace('@@RECEIPT@@', repr(pin(receipt_path)))
    compile(script, 'retire-closed-decoder-tensor-copies', 'exec'); OUT.mkdir()
    def save(name, value):
        with (OUT/name).open('x') as f: json.dump(value, f, indent=2); f.write('\n')
    save('intent.json', dict(tool=pin(Path(__file__)), model_closure=pin(MODEL/'closed.json'), receipt=pin(receipt_path), files=rows))
    with (OUT/'remote.py').open('x', newline='\n') as f: f.write(script)
    loader = importlib.util.spec_from_file_location('retirement_transport', ROOT/'tests/parakeet/ort-diagnosis-amd/run.py')
    transport = importlib.util.module_from_spec(loader); loader.loader.exec_module(transport)
    result = transport.ssh(script); save('result.json', result); assert result['passed']
    for row in rows: assert pin(MODEL/'collected'/row['name']) == row['identity']
    save('closed.json', dict(passed=True, result=pin(OUT/'result.json'), intent=pin(OUT/'intent.json'),
        script=pin(OUT/'remote.py'), locally_retained_outputs=len(rows)))
    print(json.dumps(result))


REMOTE = r'''
from pathlib import Path
import hashlib,json,os,signal,sys
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
os.sched_setaffinity(0,{0});signal.alarm(100)
def read(p):return json.loads(p.read_text())
def pin(p):
 with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
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
base=Path('/dev/shm/lokad-parakeet-decoder-packed-row-models-20260927')
assert pin(base/'collection.json')==@@RECEIPT@@
receipt=read(base/'collection.json');assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
assert not any(live(i) for i in receipt['identities'])
raw=set();documents=0
for folder in Path('/dev/shm').glob('lokad-*'):
 if not folder.is_dir():continue
 for name in ['spec.json','payload.json','stage.json']:
  p=folder/name
  if not p.is_file():continue
  value=read(p);documents+=1;raw.update(str(folder/n) for n in value.get('files',{}));raw.update(value.get('external',{}))
  for link in value.get('links',{}).values():
   if isinstance(link,dict) and isinstance(link.get('source'),str):raw.add(link['source'])
protected={str(Path(n).resolve()) for n in raw};rows=@@ROWS@@;physical=0
before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
for row in rows:
 p=base/row['name'];assert p.resolve()==p and p.is_relative_to(base) and not p.is_symlink()
 assert len(p.relative_to(base).parts)==3 and p.parent.name=='result.json.tensors'
 assert p.parts[-3] in ['selected-native-512','candidate-native-512','selected-native-256','candidate-native-256']
 assert str(p) not in protected and p.stat().st_nlink==1
 assert pin(p)==row['identity']==receipt['files'][row['name']];physical+=p.stat().st_blocks*512
for row in rows:
 p=base/row['name'];assert pin(p)==row['identity'];p.unlink()
print(json.dumps(dict(passed=True,files=len(rows),physical_bytes_freed=physical,protected_paths=len(protected),
 frozen_documents=documents,before=before,after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free))))
'''


if __name__ == '__main__': main()
