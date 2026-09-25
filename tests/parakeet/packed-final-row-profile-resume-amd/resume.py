"""Stage and collect only the unstarted phase/wall processes; retain the failed owner."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from checks import ROOT,ORIGINAL,PROFILE,APP,original_transport,partial,pin,read
from worker import adapted_capture

TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-packed-final-row-profile-resume-amd-20260925'
REMOTE=original_transport.REMOTE
PRELUDE=original_transport.PRELUDE
ssh,SSH,write=original_transport.ssh,original_transport.SSH,original_transport.write


def prepare():
    assert not BASE.exists();value=partial()
    retention=ROOT/'artifacts/parakeet-profile-headroom-retention-20260925/closed.json'
    assert read(retention)['passed'] and read(retention)['all_local_originals_retained']
    BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir()
    for old,new in [('worker.py','profile_resume.py'),('rules.py','rules.py')]:shutil.copy2(TOOLS/old,bundle/new)
    source=adapted_capture((ORIGINAL/'remote.py').read_text(encoding='utf8'))
    spec=dict(original_spec=pin(PROFILE/'bundle/spec.json'),original_worker=pin(ORIGINAL/'remote.py'),
        original_state=pin(PROFILE/'capture-collected/capture-state.json'),
        original_collection=pin(PROFILE/'capture-collected/capture-collection.json'),
        build_review=pin(PROFILE/'build-review.json'),built=pin(PROFILE/'build-collected/built.json'),
        retained_files=value['receipt']['files'],refusal=value['refusal'],
        capture_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        files={p.name:pin(p) for p in bundle.iterdir()})
    write(bundle/'resume-spec.json',spec)
    shutil.copy2(ROOT/'PLAN.md',BASE/'prospective-plan.md')
    inputs={}
    for path in TOOLS.iterdir():
        if path.is_file():
            if path.suffix=='.py':ast.parse(path.read_text(encoding='utf8'))
            inputs[path.relative_to(ROOT).as_posix()]=pin(path)
    for path in [retention,PROFILE/'capture-transfer.json',PROFILE/'capture-results.tar.gz',
                 PROFILE/'build-review-correction.json',PROFILE/'build-review.json']:
        inputs[path.relative_to(ROOT).as_posix()]=pin(path)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for path in sorted(bundle.iterdir()):archive.add(path,arcname=path.name,recursive=False)
    write(BASE/'prepared.json',dict(passed=True,inputs=inputs,archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'resume-spec.json'),
        original_prepared=pin(PROFILE/'prepared.json'),control_requests=80,control_resources=value['resources']))
    print(json.dumps(dict(passed=True,archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'resume-spec.json'),refusal=value['refusal'])))


def prepared():
    value=read(BASE/'prepared.json');assert value['passed']
    for name,wanted in value['inputs'].items():assert pin(ROOT/name)==wanted,name
    assert pin(BASE/'payload.tar.gz')==value['archive'] and pin(BASE/'bundle/resume-spec.json')==value['spec']
    assert pin(PROFILE/'prepared.json')==value['original_prepared']
    original_transport.prepared()
    return value


def stage():
    value=prepared();assert not (BASE/'staged.json').exists()
    ssh(PRELUDE+'''
from remote import idle
idle()
for name in ['resume-transfer.tar.gz','resume-spec.json','profile_resume.py','rules.py','resume-state.json']:
 assert not (base/name).exists(),name
print(json.dumps(dict(passed=True)))
''')
    subprocess.run(['scp',*SSH[1:-1],str(BASE/'payload.tar.gz'),SSH[-1]+':'+REMOTE+'/resume-transfer.tar.gz'],
        check=True,timeout=90,creationflags=subprocess.CREATE_NO_WINDOW)
    result=ssh(PRELUDE+f'''
from remote import idle,pin
idle();archive=base/'resume-transfer.tar.gz'
assert pin(archive)=={value['archive']!r}
with tarfile.open(archive) as tar:
 members=tar.getmembers()
 assert len(members)==3 and {{m.name for m in members}}=={{'resume-spec.json','profile_resume.py','rules.py'}}
 assert all(m.isfile() and not (base/m.name).exists() for m in members)
 tar.extractall(base,filter='data')
import profile_resume
profile_resume.verify()
assert pin(base/'resume-spec.json')=={value['spec']!r}
archive.unlink()
print(json.dumps(dict(passed=True,spec=pin(base/'resume-spec.json'))))
''')
    write(BASE/'staged.json',result);print(json.dumps(result))


def launch():
    value=prepared();assert read(BASE/'staged.json')['passed'] and not (BASE/'deployment.json').exists()
    result=ssh(PRELUDE+f'''
import profile_resume
from remote import idle,pin
idle();profile_resume.verify()
assert pin(base/'resume-spec.json')=={value['spec']!r} and not (base/'resume-state.json').exists()
env=dict(os.environ,PYTHONPATH={original_transport.transport.SITE!r},PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONOPTIMIZE',None)
with (base/'resume-supervisor.stdout').open('x') as out,(base/'resume-supervisor.stderr').open('x') as err:
 p=subprocess.Popen([sys.executable,'-B',str(base/'profile_resume.py')],cwd=base,env=env,
  stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
 value=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time())
print(json.dumps(value))
''')
    write(BASE/'deployment.json',result);print(json.dumps(result))


def observe():
    assert not (BASE/'closed.json').exists()
    result=ssh(PRELUDE+f'''
from remote import read,live
state=read(base/'resume-state.json') if (base/'resume-state.json').exists() else None
ids=[{read(BASE/'deployment.json')!r}]
if state:ids += [dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
print(json.dumps(dict(state=state,live=[i for i in ids if live(i)],stderr=(base/'resume-supervisor.stderr').read_text()[-4000:])))
''')
    with (BASE/'observations.jsonl').open('a') as stream:stream.write(json.dumps(result)+'\n')
    state=result['state'];latest=None if not state or not state['runs'] else {k:state['runs'][-1].get(k) for k in ['name','samples','complete','code']}
    print(json.dumps(dict(live=result['live'],complete=state and state['complete'],code=state and state['code'],
                         latest=latest,error=state and state.get('error'),stderr=result['stderr'])))


def collect():
    prepared();assert not (BASE/'collected').exists() and not (BASE/'results.tar.gz').exists()
    script=PRELUDE+'''
import profile_resume
from remote import read,live,pin
profile_resume.verify();state=read(base/'resume-state.json');previous=read(base/'capture-state.json')
ids=[s['supervisor'] for s in [previous,state]]+[dict(pid=int(p),birth=b) for s in [previous,state] for r in s['runs'] for p,b in r['members'].items()]
assert state['complete'] and not any(live(i) for i in ids)
assert not (base/'resume-collection.json').exists()
paths=[p for p in base.rglob('*') if p.is_file() and ('logs' in p.parts or p.parent==base or p.parent.name in ['control','phase','wall']) and p.name!='transfer.tar.gz']
files={p.relative_to(base).as_posix():pin(p) for p in paths}
(base/'resume-collection.json').write_text(json.dumps(dict(files=files,state=pin(base/'resume-state.json'),terminal=True,code=state['code'],identities=ids)))
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for name in [*files,'resume-collection.json']:tar.add(base/name,arcname=name,recursive=False)
'''
    with (BASE/'results.tar.gz').open('xb') as out,(BASE/'collection.stderr').open('x') as err:
        result=subprocess.run(SSH+['python3','-B','-'],input=script.encode(),stdout=out,stderr=err,timeout=180,creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode==0,'Preserve partial collection; never relaunch on collection failure'
    target=BASE/'collected';target.mkdir()
    with tarfile.open(BASE/'results.tar.gz') as tar:
        members=tar.getmembers()
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        assert len({m.name for m in members})==len(members)
        tar.extractall(target,filter='data')
    receipt=read(target/'resume-collection.json')
    for name,wanted in receipt['files'].items():assert pin(target/name)==wanted,name
    write(BASE/'transfer.json',dict(passed=True,archive=pin(BASE/'results.tar.gz'),collection=pin(target/'resume-collection.json')))
    print(json.dumps(dict(code=receipt['code'],files=len(receipt['files']))))


if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ['prepare','stage','launch','observe','collect']
    globals()[sys.argv[1]]()
