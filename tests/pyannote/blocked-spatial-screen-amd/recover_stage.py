"""Restore the unchanged payload after logout cleanup; no worker was launched."""
import json
import shutil
import subprocess
from pathlib import Path
from run import BASE, ROOT, REMOTE, SSH, PRELUDE, prepared, ssh
from protocol import pin, read, save


def main():
    spec = prepared()
    assert read(BASE/'staged.json')['passed'] and not (BASE/'deployment.json').exists()
    assert not (BASE/'logout-recovery.json').exists()
    shutil.copy2(Path(__file__), BASE/'recover_stage.py')
    save(BASE/'launch-failure.json', dict(complete=True, code=1, no_worker_launched=True,
        error="ModuleNotFoundError: No module named 'protocol' before deployment creation",
        observation='Read-only subsequent inspection found the entire remote directory absent, /dev/shm empty, and unchanged boot time.',
        hypothesis='logind RemoveIPC=yes and Linger=no with more than the default ten seconds between sessions; no causal syscall trace retained.'))
    before = json.loads(ssh(PRELUDE+'''
assert not base.exists() and psutil.boot_time()==1789634288.0
assert not any(p.info['name'] in ['dotnet','perf'] for p in psutil.process_iter(['name']))
def run(command):
 result=subprocess.run(command,text=True,capture_output=True);assert result.returncode==0,result.stderr;return result.stdout
linger=run(['loginctl','show-user','vermorel','-p','Linger','--value']).strip();assert linger=='no'
config=run(['systemd-analyze','cat-config','systemd/logind.conf'])
log=run(['sudo','-n','journalctl','-u','systemd-logind.service','--since','2026-09-22 10:45:00','--no-pager','-n','90'])
run(['sudo','-n','loginctl','enable-linger','vermorel'])
assert run(['loginctl','show-user','vermorel','-p','Linger','--value']).strip()=='yes'
print(json.dumps(dict(passed=True,previous_linger=linger,current_linger='yes',user='vermorel',boot=psutil.boot_time(),config=config,log=log,
 restoration=['sudo','-n','loginctl','disable-linger','vermorel'],restore_when='Exclusive benchmark window ends, after all workers and collection terminate.')))
'''))
    save(BASE/'temporary-linger.json', before)
    first = json.loads(ssh(PRELUDE+'''
assert not base.exists() and psutil.boot_time()==1789634288.0
assert psutil.virtual_memory().available>=12*1024**3 and psutil.disk_usage('/dev/shm').free>=3*1024**3
base.mkdir()
print(json.dumps(dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage(base).free)))
'''))
    save(BASE/'restage-started.json', first)
    subprocess.run(['scp', *SSH[1:-1], str(BASE/'payload.tar.gz'), SSH[-1]+':'+REMOTE+'/transfer.tar.gz'],
        check=True, timeout=300, creationflags=subprocess.CREATE_NO_WINDOW)
    receipt = json.loads(ssh(PRELUDE+f'''
import hashlib,importlib
archive=base/'transfer.tar.gz'
with archive.open('rb') as f:actual=dict(bytes=archive.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
assert actual=={spec['archive']!r}
with tarfile.open(archive) as tar:
 members=tar.getmembers()
 assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
 assert len({{m.name for m in members}})==len(members)
 tar.extractall(base,filter='data')
importlib.invalidate_caches()
from protocol import pin,save,verify
import remote
remote.idle();value=verify(base);assert not remote.live(value['previous_owner'])
assert pin(base/'payload.json')=={spec['payload']!r}
receipt=dict(passed=True,payload=pin(base/'payload.json'),files=len(value['files']),external=len(value['external']))
save(base/'staged.json',receipt);print(json.dumps(receipt))
''', 300))
    assert receipt == read(BASE/'staged.json')
    save(BASE/'restaged.json', receipt)
    save(BASE/'logout-recovery.json', dict(passed=True, no_worker_before_recovery=True,
        original_payload=spec['payload'], original_archive=spec['archive'], staged=receipt,
        files={p.relative_to(BASE).as_posix(): pin(p) for p in [BASE/'recover_stage.py', BASE/'launch-failure.json', BASE/'temporary-linger.json', BASE/'restage-started.json', BASE/'restaged.json']}))
    print(json.dumps(dict(passed=True, payload=spec['payload'], previous_linger='no', temporary_linger='yes')))


if __name__ == '__main__': main()
