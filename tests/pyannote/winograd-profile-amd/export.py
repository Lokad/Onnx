"""Export on AMD, observe exact owners, then collect only terminal evidence."""
import subprocess
import sys
import tarfile
from common import *
from transport import PRELUDE, ssh


def launch():
    prepared(); assert not (BASE/'export-deployment.json').exists()
    capture=read(BASE/'collected/collection.json')
    assert capture['terminal'] and capture['code']==0 and capture['input_error'] is None
    receipt=json.loads(ssh(PRELUDE+f'''
assert pin(base/'collection.json')=={pin(BASE/'collected/collection.json')!r}
assert not (base/'exports').exists() and not (base/'export-deployment.json').exists()
sys.path.insert(0,str(base/'tools'));import remote
remote.idle();remote.verify()
for i in read(base/'collection.json')['identities']:assert not live(i)
with (base/'export-supervisor.stdout').open('x') as out,(base/'export-supervisor.stderr').open('x') as err:
 p=subprocess.Popen([sys.executable,'-B',str(base/'tools/remote_export.py')],cwd=base,env=remote.clean_env(),stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
 receipt=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time())
write(base/'export-deployment.json',receipt);print(json.dumps(receipt))
''',300))
    save(BASE/'export-deployment.json',receipt);print(json.dumps(receipt))


def observe():
    value=json.loads(ssh(PRELUDE+'''
deployment=read(base/'export-deployment.json')
state=read(base/'exports/identity.json') if (base/'exports/identity.json').exists() else None
ids=[deployment]+([] if state is None else [r['identity'] for r in state['runs'] if 'identity' in r])
print(json.dumps(dict(live=[i for i in ids if live(i)],complete=False if state is None else state['complete'],code=None if state is None else state['code'],latest=None if state is None or not state['runs'] else state['runs'][-1],stderr=(base/'export-supervisor.stderr').read_text()[-5000:],stdout=(base/'export-supervisor.stdout').read_text()[-2000:])))
'''))
    with (BASE/'export-observations.jsonl').open('a',encoding='utf8') as out:out.write(json.dumps(value)+'\n')
    print(json.dumps(value))


def collect():
    prepared();assert not (BASE/'exports').exists() and not (BASE/'export-results.tar.gz').exists()
    script=PRELUDE+'''
deployment=read(base/'export-deployment.json');assert not live(deployment)
state=read(base/'exports/identity.json');assert state['complete']
ids=[deployment]+[r['identity'] for r in state['runs'] if 'identity' in r]
assert all(r['complete'] for r in state['runs']) and not any(live(i) for i in ids)
sys.path.insert(0,str(base/'tools'));import remote
error=None
try:
 remote.verify()
 for name,wanted in read(base/'collection.json')['files'].items():assert pin(base/name)==wanted,name
except BaseException as e:error=repr(e)
for name in ['export-supervisor.stdout','export-supervisor.stderr','export-deployment.json']:
 shutil.copy2(base/name,base/'exports'/name)
files={p.relative_to(base/'exports').as_posix():pin(p) for p in sorted((base/'exports').rglob('*')) if p.is_file()}
receipt=dict(terminal=True,identities=ids,code=state['code'],input_error=error,files=files,payload=pin(base/'payload.json'),capture_receipt=pin(base/'collection.json'))
write(base/'exports/collection.json',receipt)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for name in [*files,'collection.json']:tar.add(base/'exports'/name,arcname=name,recursive=False)
'''
    compile(script,'stream-current-profile-exports','exec')
    with (BASE/'export-results.tar.gz').open('xb') as out,(BASE/'export-collection.stderr').open('x') as err:
        result=subprocess.run(SSH+['python3 -B -'],input=script.encode(),stdout=out,stderr=err,timeout=600,creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode==0,'Retain failed collection and partial archive'
    target=BASE/'exports';target.mkdir()
    with tarfile.open(BASE/'export-results.tar.gz') as tar:
        members=tar.getmembers()
        assert all(m.isfile() and (target/m.name).resolve().is_relative_to(target) for m in members)
        assert len({m.name for m in members})==len(members)
        tar.extractall(target,filter='data')
    receipt=read(target/'collection.json')
    for name,wanted in receipt['files'].items():assert pin(target/name)==wanted,name
    assert receipt['payload']==pin(BASE/'payload/payload.json') and receipt['capture_receipt']==pin(BASE/'collected/collection.json')
    assert {p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file()}==set(receipt['files'])|{'collection.json'}
    save(BASE/'export-transfer.json',dict(archive=pin(BASE/'export-results.tar.gz'),receipt=pin(target/'collection.json'),files=len(receipt['files']),terminal=True,code=receipt['code'],input_error=receipt['input_error']))
    print(json.dumps(read(BASE/'export-transfer.json')))


if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ['launch','observe','collect']
    globals()[sys.argv[1]]()
