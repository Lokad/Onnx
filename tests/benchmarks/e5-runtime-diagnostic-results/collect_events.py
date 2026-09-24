"""Incrementally retain completed event exports; never touch captures or active output."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/benchmarks/e5-runtime-diagnostic-amd'))
from run import BASE,PRELUDE,SSH,ssh
from protocol import pin,read,save,ROLES
OUT=ROOT/'artifacts/e5-runtime-event-transfers-20260924'
SUPERVISOR=dict(pid=954546,birth=1790264329.59)


def raw_identity(path):
    digest=hashlib.sha256();size=0
    with gzip.open(path,'rb') as stream:
        while chunk:=stream.read(1024**2):size+=len(chunk);digest.update(chunk)
    return dict(bytes=size,sha256=digest.hexdigest())


def prelude(role):
    assert role in ROLES
    return PRELUDE+f'''
from protocol import read,pin
from remote import live
state=read(base/'identity.json');assert state['supervisor']=={SUPERVISOR!r}
captures=[r for r in state['runs'] if r['name'].endswith('-capture')]
assert [r['name'] for r in captures]==['a-capture','b-capture','c-capture','d-capture']
assert all(r['complete'] and r['code']==0 for r in captures)
row,=[r for r in state['runs'] if r['name']=={role+'-export'!r}]
assert row['complete'] and row['code']==0
assert all(not live(dict(pid=int(p),birth=b)) for p,b in row['members'].items())
target=base/{role+'-export/events/events.jsonl'!r}
assert target.resolve().is_relative_to(base.resolve()) and not target.is_symlink() and target.stat().st_nlink==1
summary=read(target.parent/'summary.json');assert summary['complete'] and summary['lost']==0
assert summary['input_sha256']==pin(base/{role+'-capture/capture.nettrace'!r})['sha256']
'''


def transfer(role):
    OUT.mkdir(exist_ok=True);receipt=OUT/f'{role}.json';archive=OUT/f'{role}.jsonl.gz'
    assert not receipt.exists() and not archive.exists()
    info=json.loads(ssh(prelude(role)+"print(json.dumps(dict(identity=pin(target),summary=pin(target.parent/'summary.json'),events=summary['recorded'])))\n"))
    script=prelude(role)+f'''
import gzip,shutil
assert pin(target)=={info['identity']!r}
with target.open('rb') as source,gzip.GzipFile(fileobj=sys.stdout.buffer,mode='wb',mtime=0) as output:
 shutil.copyfileobj(source,output,1024**2)
assert pin(target)=={info['identity']!r}
'''
    compile(script,'e5-event-transfer','exec')
    with archive.open('xb') as stream:
        result=subprocess.run(SSH+['python3 -B -'],input=script.encode(),stdout=stream,stderr=subprocess.PIPE,
            timeout=180,creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode==0,result.stderr.decode()[-4000:]
    assert raw_identity(archive)==info['identity']
    save(OUT/f'{role}-received.json',dict(passed=True,role=role,archive=pin(archive),**info))
    retired=json.loads(ssh(prelude(role)+f'''
assert pin(target)=={info['identity']!r}
# Export text is never an input to another job. Preserve raw trace and summary.
spec=read(base/'payload.json');assert target.relative_to(base).as_posix() not in spec['files']
assert str(target) not in spec['external']
before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage(base).free)
target.unlink();assert not target.exists()
print(json.dumps(dict(passed=True,retired=str(target),identity={info['identity']!r},before=before,
 after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage(base).free))))
'''))
    save(receipt,dict(passed=True,role=role,archive=pin(archive),raw=info['identity'],summary=info['summary'],
        events=info['events'],retirement=retired,source=pin(Path(__file__)),received=pin(OUT/f'{role}-received.json')))
    print(json.dumps(dict(role=role,bytes_retired=info['identity']['bytes'],archive=pin(archive))))


def restore():
    folder=BASE/'collected';assert (folder/'collection.json').exists()
    collection=read(folder/'collection.json');assert collection['terminal'] and collection['code']==0
    restored={}
    for role in ROLES:
        receipt=read(OUT/f'{role}.json');archive=OUT/f'{role}.jsonl.gz'
        assert receipt['passed'] and pin(archive)==receipt['archive'] and raw_identity(archive)==receipt['raw']
        name=f'{role}-export/events/events.jsonl';target=folder/name
        assert name not in collection['files'] and not target.exists()
        assert pin(target.parent/'summary.json')==receipt['summary']
        with gzip.open(archive,'rb') as source,target.open('xb') as output:shutil.copyfileobj(source,output,1024**2)
        assert pin(target)==receipt['raw']
        restored[name]=dict(identity=pin(target),receipt=pin(OUT/f'{role}.json'),archive=pin(archive))
    save(folder/'incremental-event-collection.json',dict(passed=True,files=restored,
        source=pin(Path(__file__)),original_collection=pin(folder/'collection.json')))
    print(json.dumps(dict(passed=True,files=len(restored),bytes=sum(r['identity']['bytes'] for r in restored.values()))))


if __name__=='__main__':
    assert len(sys.argv)==2
    if sys.argv[1]=='restore':restore()
    else:transfer(sys.argv[1])
