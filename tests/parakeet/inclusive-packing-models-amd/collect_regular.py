"""Collect hardlinked model evidence as regular members; all gates stay unchanged."""
import inspect
import subprocess
import sys
import tarfile
from pathlib import Path
import run
from run import BASE, PRELUDE, SSH, prepared
from protocol import pin, read, save


def collect():
    prepared(); assert not (BASE/'collected').exists() and not (BASE/'results.tar.gz').exists()
    script = PRELUDE+'''
from protocol import JOBS,pin,read,save,verify
from remote import live
deployment=read(base/'deployment.json');state=read(base/'identity.json')
ids=[deployment]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
assert state['complete'] and not any(live(i) for i in ids)
error=None
try:verify(base)
except BaseException as e:error=repr(e)
paths=[p for folder in [*JOBS,'logs','evidence','manifests','parakeet-reference','assets','runtimes','products'] for p in (base/folder).rglob('*') if p.is_file()]
paths += [p for p in base.iterdir() if p.is_file() and p.name!='transfer.tar.gz']
files={p.relative_to(base).as_posix():pin(p) for p in sorted(paths)}
save(base/'collection.json',dict(terminal=True,identities=ids,code=state['code'],input_error=error,files=files,payload=pin(base/'payload.json')))
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz',dereference=True) as tar:
 for name in [*files,'collection.json']:tar.add(base/name,arcname=name,recursive=False)
'''
    compile(script,'native-layout-collect','exec')
    with (BASE/'results.tar.gz').open('xb') as out,(BASE/'collection.stderr').open('x') as err:
        result = subprocess.run(SSH+['python3 -B -'],input=script.encode(),stdout=out,stderr=err,timeout=600,creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode == 0,'Preserve partial collection; do not relaunch'
    target = BASE/'collected'; target.mkdir()
    with tarfile.open(BASE/'results.tar.gz') as tar:
        members = tar.getmembers()
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        assert len({m.name for m in members}) == len(members)
        tar.extractall(target,filter='data')
    receipt = read(target/'collection.json')
    for name,wanted in receipt['files'].items(): assert pin(target/name) == wanted,name
    assert receipt['payload']==pin(BASE/'payload.json')
    save(BASE/'collection-transfer.json',dict(passed=True,archive=pin(BASE/'results.tar.gz'),receipt=pin(target/'collection.json')))
    print(json.dumps(dict(code=receipt['code'],terminal=receipt['terminal'],files=len(receipt['files']))))


if __name__=='__main__':
    before="with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:"
    after="with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz',dereference=True) as tar:"
    original=inspect.getsource(run.collect)
    assert original.count(before)==1
    assert inspect.getsource(collect)==original.replace(before,after)
    prepared()
    receipt=BASE/'collection-transport.json'
    assert not receipt.exists()
    save(receipt,dict(passed=True,original=pin(Path(run.__file__)),collector=pin(Path(__file__)),
        sole_transport_change='dereference=True: export each hardlinked immutable file as a regular member; exact manifest and safe extraction unchanged'))
    collect()
