"""Retire one verified VM transfer duplicate between workers; retain its local archive."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/parakeet/validated-composition-graphs-amd'))
from run import ssh, PRELUDE
from protocol import pin, read, save

SOURCE = ROOT/'artifacts/parakeet-selected-profile-amd-20260924'
OUT = ROOT/'artifacts/parakeet-profile-transfer-retention-20260924'
REMOTE = '/dev/shm/lokad-parakeet-selected-profile-20260924'
SUPERVISOR = dict(pid=948414,birth=1790257660.39)


def main():
    assert not OUT.exists()
    proof = read(SOURCE/'closed.json')
    assert proof['passed'] and pin(SOURCE/'closed.json')['sha256'] == 'e6a94afd464807f625e439bdba4a4b71246e641e98b10a632b66bb068adc5937'
    archive = SOURCE/'payload.tar.gz'; wanted = pin(archive)
    assert wanted == proof['files'][archive.relative_to(ROOT).as_posix()]
    assert wanted['bytes'] == 20313134
    script = PRELUDE+f'''
from protocol import pin,read
from remote import live
assert psutil.boot_time()==1789634288.0
def between_workers():
 state=read(base/'identity.json'); assert state['supervisor']=={SUPERVISOR!r}
 assert all(r['complete'] and r['code']==0 for r in state['runs'])
 assert all(not live(dict(pid=int(p),birth=b)) for r in state['runs'] for p,b in r['members'].items())
 own=psutil.Process();allowed={{own.pid,*[p.pid for p in own.parents()],{SUPERVISOR['pid']}}}
 for p in psutil.process_iter(['name','cmdline']):
  if p.pid in allowed:continue
  command=' '.join(p.info['cmdline'] or [])
  assert p.info['name'] not in ['dotnet','perf'],p.info
  assert not (p.info['name'].startswith('python') and '/dev/shm/lokad-' in command),p.info
 return dict(supervisor_live=live(state['supervisor']),completed=len(state['runs']),complete=state['complete'])
first=between_workers()
root=Path({REMOTE!r});target=root/'transfer.tar.gz'
assert root.resolve().parent==Path('/dev/shm') and target.resolve()==target
assert not target.is_symlink() and target.stat().st_nlink==1
assert not any(live(i) for i in {proof['remote_identities']!r})
protected=set()
for folder in Path('/dev/shm').glob('lokad-*'):
 for name in ['payload.json','stage.json','spec.json']:
  if not (folder/name).exists():continue
  value=read(folder/name)
  protected.update(str((folder/n).resolve()) for n in value.get('files',{{}}))
  protected.update(str(Path(n).resolve()) for n in value.get('external',{{}}))
assert str(target) not in protected and pin(target)=={wanted!r}
before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
last=between_workers(); assert last==first
target.unlink();assert not target.exists()
print(json.dumps(dict(passed=True,retired=str(target),bytes_retired={wanted['bytes']},identity={wanted!r},
 before=before,after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free),
 protected_paths=len(protected),between_workers=last)))
'''
    result = json.loads(ssh(script))
    assert result['passed'] and pin(archive) == wanted
    OUT.mkdir()
    save(OUT/'closed.json',dict(**result,local_archive_retained=archive.relative_to(ROOT).as_posix(),
        source_closure=pin(SOURCE/'closed.json'),generator=pin(Path(__file__))))
    print(json.dumps(result))


if __name__ == '__main__': main()
