"""Retire three sets of closed VM output duplicates, keeping verified local arrays."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/parakeet-decoder-graph-headroom-v2-20260927'
FAILED = ROOT/'artifacts/parakeet-decoder-graph-headroom-20260927'
CAMPAIGNS = [
    ('parakeet-decoder-packed-row-pyannote-amd-20260927', 'lokad-parakeet-decoder-packed-row-pyannote-20260927',
     '74bd1f90716a11446846d12cbba344b501d2736777ac1d0b3eedd6eb10ca7a33', 'pyannote', 48),
    ('parakeet-decoder-packed-row-shared-amd-20260927', 'lokad-parakeet-decoder-packed-row-shared-20260927',
     'bdd1c7664d902af7d135318f12856419a61d222e69bb073ae24e10b8aca8aca0', 'shared', 332),
    ('parakeet-rational-sigmoid-pyannote-amd-20260927', 'lokad-parakeet-rational-sigmoid-pyannote-20260927',
     '41333fc2a2c60c45596ebb87d65c4ce5fdfa23108ed2064a2db5e308dc9e5138', 'pyannote', 48),
]


def read(path): return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def selected(name, kind):
    parts = Path(name).parts
    if len(parts) != 3: return False
    roles = ['selected', 'candidate'] if kind == 'pyannote' else [r+'-'+m for r in ['selected', 'candidate'] for m in ['shared', 'e5']]
    return parts[0] in roles and parts[1] == 'output' and parts[2].endswith('.f32')


def main():
    assert not OUT.exists(), 'One-time retirement; preserve any partial attempt'
    failure = read(FAILED/'failed.json')
    assert failure['terminal'] and failure['no_deletion'] and not failure['passed']
    for name, wanted in failure['files'].items(): assert pin(FAILED/name) == wanted
    assert pin(FAILED/'failed.json')['sha256'] == '82992a27f3cccb48c6df3a901c14b1de3abea2c7562eccb6910ab73eb5f1bfcc'
    groups = []
    for local, remote, digest, kind, count in CAMPAIGNS:
        base = ROOT/'artifacts'/local
        assert pin(base/'closed.json')['sha256'] == digest
        proof = read(base/'closed.json'); receipt = read(base/'collected/collection.json')
        assert proof['passed'] and proof['files']['collected/collection.json'] == pin(base/'collected/collection.json')
        assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
        initial = read(base/'payload.json'); rows = []
        for name, wanted in receipt['files'].items():
            if not selected(name, kind): continue
            assert name not in initial['files']
            assert pin(base/'collected'/name) == wanted == proof['files']['collected/'+name]
            rows.append(dict(name=name, identity=wanted))
        assert len(rows) == count
        groups.append(dict(local=local, base='/dev/shm/'+remote, kind=kind, files=rows,
                           closure=pin(base/'closed.json'), receipt=pin(base/'collected/collection.json')))
    script = REMOTE.replace('@@GROUPS@@', repr(groups))
    compile(script, 'retire-closed-graph-headroom', 'exec'); OUT.mkdir()
    def save(name, value):
        with (OUT/name).open('x', encoding='utf8', newline='\n') as stream:
            json.dump(value, stream, indent=2); stream.write('\n')
    save('intent.json', dict(tool=pin(Path(__file__)), groups=groups, expected_physical_bytes=98713600, preserved_failure=pin(FAILED/'failed.json')))
    with (OUT/'remote.py').open('x', encoding='utf8', newline='\n') as stream: stream.write(script)
    loader = importlib.util.spec_from_file_location('retirement_transport', ROOT/'tests/parakeet/ort-diagnosis-amd/run.py')
    transport = importlib.util.module_from_spec(loader); loader.loader.exec_module(transport)
    result = transport.ssh(script); save('result.json', result); assert result['passed']
    for group in groups:
        for row in group['files']:
            assert pin(ROOT/'artifacts'/group['local']/'collected'/row['name']) == row['identity']
    save('closed.json', dict(passed=True, result=pin(OUT/'result.json'), intent=pin(OUT/'intent.json'),
                            script=pin(OUT/'remote.py'), locally_retained_outputs=428))
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
groups=@@GROUPS@@;raw=set();documents=0
for folder in Path('/dev/shm').glob('lokad-*'):
 if not folder.is_dir():continue
 for name in ['spec.json','payload.json','stage.json']:
  p=folder/name
  if not p.is_file():continue
  value=read(p);documents+=1;raw.update(str(folder/n) for n in value.get('files',{}));raw.update(value.get('external',{}))
  for link in value.get('links',{}).values():
   if isinstance(link,dict) and isinstance(link.get('source'),str):raw.add(link['source'])
protected={str(Path(n).resolve()) for n in raw};paths=[];physical=0
before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
for group in groups:
 base=Path(group['base']);assert base.parent==Path('/dev/shm') and base.name.startswith('lokad-parakeet-')
 assert pin(base/'collection.json')==group['receipt']
 receipt=read(base/'collection.json');assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
 assert not any(live(i) for i in receipt['identities'])
 initial=read(base/'payload.json')
 for row in group['files']:
  p=base/row['name'];assert p.resolve()==p and p.is_relative_to(base) and not p.is_symlink()
  parts=p.relative_to(base).parts;assert len(parts)==3
  roles=['selected','candidate'] if group['kind']=='pyannote' else ['selected-shared','selected-e5','candidate-shared','candidate-e5']
  assert parts[0] in roles and parts[1]=='output' and p.suffix=='.f32'
  assert row['name'] not in initial['files'] and str(p) not in protected and p.stat().st_nlink==1
  assert pin(p)==row['identity']==receipt['files'][row['name']]
  physical+=p.stat().st_blocks*512;paths.append((p,row['identity']))
assert len(paths)==428 and physical==98713600
for p,wanted in paths:
 assert pin(p)==wanted;p.unlink()
print(json.dumps(dict(passed=True,files=len(paths),physical_bytes_freed=physical,protected_paths=len(protected),
 frozen_documents=documents,before=before,after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free))))
'''


if __name__ == '__main__': main()
