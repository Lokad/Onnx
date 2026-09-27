"""Retire only closed model-output VM copies after verifying retained local evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/parakeet-rational-pyannote-headroom-20260927'
SOURCES = [
    ('parakeet-rational-sigmoid-models-amd-20260927','lokad-parakeet-rational-sigmoid-models-20260927',
     '3d2e01b5aca3f66c434d5d1e8124c2b0d08327bc14dd37409c0e1884619769d8',
     [role+'-native-'+mode+'/result.json.tensors' for role in ['selected','candidate'] for mode in ['512','256']], '.bin', 3136),
    ('parakeet-rational-sigmoid-shared-amd-20260927','lokad-parakeet-rational-sigmoid-shared-20260927',
     '86ad6b3f5d11549dc968270e6ef5936787a266a84969de1932e750450ee6da65',
     [role+'-'+mode+'/output' for role in ['selected','candidate'] for mode in ['shared','e5']], '.f32', 332),
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
        proof = json.loads((base/'closed.json').read_text()); assert proof['passed']
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
assert len(checked)==3468 and len({r['path'] for r in checked})==3468
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
