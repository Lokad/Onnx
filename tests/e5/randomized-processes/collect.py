"""Audit a terminal phase on its host, then transfer all retained evidence."""
import argparse
import json
import shlex
import subprocess
import tarfile

from contract import pin, read, write
from remote import BASE, REMOTE, PRELUDE, SSH, ssh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['aa', 'compare'], required=True)
    args = parser.parse_args(); phase = args.phase
    target = BASE/('collected-'+phase); archive_path = BASE/('results-'+phase+'.tar.gz')
    assert not target.exists() and not archive_path.exists()
    frozen = pin(BASE/'frozen.json')
    # The raw auditor runs once. If an earlier transfer failed, a terminal saved
    # audit is reused rather than silently launching it again.
    script = PRELUDE+'''
os.sched_setaffinity(0,{0});phase=%r
assert pin(base/'frozen.json')==%r
meta=read(base/'frozen.json');state=read(base/('result-'+phase)/'identity.json')
assert state['complete'] is True
births=[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
terminal(births)
audit_path=base/(phase+'-remote-audit.json');process_path=base/(phase+'-audit-process.json')
if state['code']==0:
 if process_path.exists():
  process=read(process_path);assert process['complete'] is True;terminal([process['child']])
 else:
  assert not audit_path.exists()
  env=dict(os.environ,PYTHONPATH=os.pathsep.join(meta['python_paths']),PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1')
  for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='1'
  with (base/(phase+'-audit.stdout')).open('x') as out,(base/(phase+'-audit.stderr')).open('x') as err:
   p=subprocess.Popen([meta['python_executable'],'-B',str(base/'tests/e5/randomized-processes/contract.py'),'--payload',str(base),'--phase',phase,'--output',str(audit_path)],cwd=base,env=env,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True)
   process=dict(child=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time()),complete=False,started=time.time())
   write(process_path,process)
   code=p.wait(timeout=1800);process.update(code=code,complete=True,ended=time.time())
   temporary=process_path.with_suffix('.tmp');temporary.write_text(json.dumps(process,indent=2));temporary.replace(process_path)
  terminal([process['child']])
 births.append(process['child'])
else:process=None
collection_path=base/(phase+'-collection.json')
if collection_path.exists():
 collection=read(collection_path);assert collection['frozen']==pin(base/'frozen.json');terminal(collection['births'])
 for name,wanted in collection['files'].items():assert pin(base/name)==wanted,name
else:
 external={item['path']:{k:item[k] for k in ['bytes','sha256']} for item in [meta['model'],meta['native'],meta['dotnet']]+meta['runtime_files']}
 external.update(meta['python_files']);external[meta['python_executable']]=meta['interpreter']
 for path,wanted in external.items():assert pin(path)==wanted,path
 names=set(meta['files'])|{'frozen.json','deployment-'+phase+'.json','supervisor-'+phase+'.stdout','supervisor-'+phase+'.stderr'}
 names.update(p.relative_to(base).as_posix() for p in (base/('result-'+phase)).rglob('*') if p.is_file())
 for name in [phase+'-remote-audit.json',phase+'-audit-process.json',phase+'-audit.stdout',phase+'-audit.stderr']:
  if (base/name).exists():names.add(name)
 if phase=='compare':names.add('aa-gate.json')
 files={name:pin(base/name) for name in sorted(names)}
 collection=dict(terminal=True,phase=phase,code=state['code'],frozen=pin(base/'frozen.json'),births=births,external=external,files=files,
  remote_audit_code=None if process is None else process['code'])
 write(collection_path,collection)
print(json.dumps(dict(receipt=pin(collection_path),files=len(collection['files']),births=collection['births'],code=collection['code'],remote_audit_code=collection['remote_audit_code'])))
''' % (phase, frozen)
    snapshot = json.loads(ssh(script, timeout=1900))
    write(BASE/('collection-snapshot-'+phase+'.json'), snapshot)
    # Stream the archive directly to local disk, avoiding a second ~GB copy on
    # the VM's tmpfs. The immutable inventory independently verifies every file.
    script = PRELUDE+'''
import tarfile
phase=%r;receipt=read(base/(phase+'-collection.json'));terminal(receipt['births'])
assert pin(base/(phase+'-collection.json'))==%r
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for name in list(receipt['files'])+[phase+'-collection.json']:
  archive.add(base/name,arcname=name,recursive=False)
''' % (phase, snapshot['receipt'])
    compile(script, 'checked-e5-archive-stream', 'exec')
    with archive_path.open('xb') as output, (BASE/('transfer-'+phase+'.stderr')).open('x') as error:
        result = subprocess.run(SSH+['python3 -B -'], input=script.encode('utf8'), stdout=output, stderr=error,
                                timeout=1800, creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode == 0
    target.mkdir()
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        assert all(m.isfile() and not m.name.startswith('/') and '..' not in m.name.split('/') for m in members)
        assert len({m.name for m in members}) == len(members)
        archive.extractall(target, filter='data')
    receipt = target/(phase+'-collection.json')
    assert pin(receipt) == snapshot['receipt']
    inventory = read(receipt)
    for name, expected in inventory['files'].items():
        assert pin(target/name) == expected, name
    assert {p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file()} == set(inventory['files']) | {receipt.name}
    assert pin(target/'frozen.json') == frozen
    # Verify terminal identities again after the transfer; elapsed time alone
    # would not establish termination.
    assert json.loads(ssh(PRELUDE+'terminal(%r)\nprint(json.dumps(dict(terminal=True)))\n' % inventory['births']))['terminal']
    write(BASE/('collection-transfer-'+phase+'.json'), dict(passed=True, archive=pin(archive_path), receipt=pin(receipt),
          files=len(inventory['files']), births=inventory['births'], remote_audit_code=inventory['remote_audit_code']))
    print(json.dumps(dict(collected=True, phase=phase, files=len(inventory['files']), code=inventory['code'], remote_audit_code=inventory['remote_audit_code'])))


if __name__ == '__main__':
    main()
