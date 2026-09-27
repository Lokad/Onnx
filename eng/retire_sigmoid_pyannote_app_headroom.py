"""Retire rational graph-output VM copies before the Pyannote application check."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/parakeet-rational-pyannote-app-headroom-20260927'
CASES = ['e5-8tok','e5-30tok','e5-30pad128','e5-128tok','e5-512tok','dinov3','resnet50','gpt2']
ORDER = ['current-a','candidate-a','ort-a','ort-b','candidate-b','current-b']
JOBS = [f'verify-{case}-{role}' for case in CASES for role in ['current','candidate','ort']] + [f'timing-{case}-{role}' for case in CASES for role in ORDER]
SOURCES = [
    ('parakeet-rational-sigmoid-graphs-amd-20260927','lokad-parakeet-rational-sigmoid-graphs-20260927',
     '167f01fdd3c8dce104377259ae4441a8e59c0a6bd6d666e457d38e3398153742',
     ['logs']+[name+'/output' for name in JOBS], '', 585),
]


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def main():
    assert not OUT.exists(), 'One-time retirement already attempted'
    specs = []
    for local,remote,digest,parents,suffix,count in SOURCES:
        base = ROOT/'artifacts'/local
        assert pin(base/'closed.json')['sha256'] == digest
        proof = json.loads((base/'closed.json').read_text()); assert proof['passed'] and proof['admitted']
        assert pin(base/'payload.json')==proof['files']['payload.json']
        assert json.loads((base/'payload.json').read_text())['jobs']==JOBS
        receipt = base/'collected/collection.json'
        assert pin(receipt) == proof['files'][receipt.relative_to(base).as_posix()]
        collected = json.loads(receipt.read_text()); assert collected['terminal'] and collected['code']==0 and collected['input_error'] is None
        files = {}
        for name,wanted in collected['files'].items():
            if Path(name).parent.as_posix() not in parents or not name.endswith(suffix):continue
            path=base/'collected'/name
            assert pin(path) == wanted == proof['files'][path.relative_to(base).as_posix()]
            files[name] = wanted
        assert len(files)==count
        specs.append(dict(base='/dev/shm/'+remote,collection=pin(receipt),files=files,allowed_parents=parents))
    loader = importlib.util.spec_from_file_location('headroom_transport',ROOT/'tests/parakeet/rational-sigmoid-build/run.py')
    module = importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
    script = '''
from pathlib import Path
import hashlib,json,os,signal,sys
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
os.sched_setaffinity(0,{0});signal.alarm(180)
def pin(p):
    with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def read(p):return json.loads(p.read_text())
def live(i):
    try:
        p=psutil.Process(i['pid']);return p.create_time()==i['birth'] and p.status()!=psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:return False
own=psutil.Process();ancestors={own.pid,*[p.pid for p in own.parents()]}
assert psutil.boot_time()==1789634288.0
for p in psutil.process_iter(['name','cmdline']):
    if p.pid in ancestors:continue
    assert p.info['name'] not in ['dotnet','perf']
    assert not (p.info['name'].startswith('python') and '/dev/shm/lokad-' in ' '.join(p.info['cmdline'] or []))
unresolved=set();documents=0
for folder in Path('/dev/shm').glob('lokad-*'):
    if not folder.is_dir():continue
    for name in ['spec.json','payload.json','stage.json']:
        path=folder/name
        if not path.is_file():continue
        spec=read(path);documents+=1
        for name in spec.get('files',{}):unresolved.add(str(folder/name))
        for name in spec.get('external',{}):unresolved.add(name)
protected={str(Path(name).resolve()) for name in unresolved}
checked=[];before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
for spec in SPECS:
    base=Path(spec['base']).resolve();assert base.parent==Path('/dev/shm')
    receipt_path=base/'collection.json';assert pin(receipt_path)==spec['collection']
    receipt=read(receipt_path)
    assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None and not any(live(i) for i in receipt['identities'])
    for name,wanted in spec['files'].items():
        path=base/name
        assert path.resolve()==path and path.is_relative_to(base) and not path.is_symlink()
        assert path.parent.relative_to(base).as_posix() in spec['allowed_parents']
        assert str(path) not in protected and pin(path)==wanted==receipt['files'][name]
        stat=path.stat();assert stat.st_nlink==1
        checked.append(dict(path=str(path),identity=wanted,physical_bytes=stat.st_blocks*512))
assert len(checked)==585 and len({r['path'] for r in checked})==585
for row in checked:
    path=Path(row['path']);assert pin(path)==row['identity'];path.unlink()
print(json.dumps(dict(passed=True,files=checked,protected_paths=len(protected),frozen_documents=documents,
    physical_bytes_freed=sum(r['physical_bytes'] for r in checked),before=before,
    after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free))))
'''.replace('SPECS',repr(specs))
    compile(script,'closed-model-output-retirement','exec')
    OUT.mkdir()
    (OUT/'intent.json').write_text(json.dumps(dict(specs=specs,tool=pin(Path(__file__))),indent=2))
    (OUT/'remote.py').write_text(script,encoding='utf8')
    result = module.ssh(script)
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    (OUT/'closed.json').write_text(json.dumps(dict(passed=result['passed'],result=pin(OUT/'result.json'),
        script=pin(OUT/'remote.py'),intent=pin(OUT/'intent.json')),indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='files'}))


if __name__ == '__main__': main()
