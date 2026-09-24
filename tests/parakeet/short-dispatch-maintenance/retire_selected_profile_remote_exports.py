"""Retire four closed VM exports; retain and verify the full local profile proof."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tests/parakeet/dense-scalar-where-build-amd'))
from run import ssh, PRELUDE
from protocol import pin, read, save

BASE = ROOT / 'artifacts/parakeet-selected-profile-remote-retention-20260924'
PROFILE = ROOT / 'artifacts/parakeet-selected-profile-amd-20260924'
REMOTE = '/dev/shm/lokad-parakeet-selected-profile-20260924'


def main():
    assert not BASE.exists()
    assert pin(PROFILE / 'closed.json')['sha256'] == 'e6a94afd464807f625e439bdba4a4b71246e641e98b10a632b66bb068adc5937'
    proof = read(PROFILE / 'closed.json'); assert proof['passed']
    # This profile's closure uses repository-relative paths, unlike build proofs.
    for name, wanted in proof['files'].items(): assert pin(ROOT / name) == wanted, name
    names = [f'{capture}/{kind}.{kind}.json' for capture in ['sampled-a', 'sampled-b']
             for kind in ['chromium', 'speedscope']]
    receipt = read(PROFILE / 'exports/collection.json')
    files = {}
    for name in names:
        kept = PROFILE / 'exports' / name
        assert pin(kept) == receipt['files'][name] == proof['files'][kept.relative_to(ROOT).as_posix()]
        files[REMOTE + '/exports/' + name] = dict(identity=pin(kept), retained=kept.relative_to(ROOT).as_posix())
    receipts = {REMOTE + '/' + remote: pin(PROFILE / local) for remote, local in [
        ('collection.json', 'collected/collection.json'), ('exports/collection.json', 'exports/collection.json')]}
    BASE.mkdir()
    preflight = PRELUDE + f'''
from protocol import read,pin
from remote import live,idle
idle(); assert psutil.boot_time()==1789634288.0
root=Path({REMOTE!r}); files={files!r}; receipts={receipts!r}
assert root.resolve().parent==Path('/dev/shm')
protected=set()
for folder in Path('/dev/shm').glob('lokad-*'):
 if not (folder/'payload.json').exists():continue
 spec=read(folder/'payload.json')
 protected.update(str((folder/n).resolve()) for n in spec.get('files',{{}}))
 protected.update(str(Path(n).resolve()) for n in spec.get('external',{{}}))
for name,wanted in receipts.items():
 assert pin(Path(name))==wanted
 value=read(Path(name))
 assert value['terminal'] and value['code']==0 and value['input_error'] is None
 assert not any(live(i) for i in value['identities'])
spec=read(root/'payload.json')
for name,wanted in spec['files'].items():assert pin(root/name)==wanted,name
for name,wanted in spec['external'].items():assert pin(Path(name))==wanted,name
for name,value in files.items():
 path=Path(name)
 assert not path.is_symlink() and str(path.resolve())==name and name not in protected
 assert path.is_relative_to(root/'exports') and pin(path)==value['identity']
'''
    snapshot = json.loads(ssh(preflight + "print(json.dumps(dict(passed=True,available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)))", 300))
    save(BASE / 'snapshot.json', dict(**snapshot, files=files, receipts=receipts, profile=pin(PROFILE / 'closed.json')))
    result = json.loads(ssh(preflight + '''
for name in files:Path(name).unlink()
assert all(not Path(name).exists() for name in files)
print(json.dumps(dict(passed=True,files=len(files),bytes=sum(v['identity']['bytes'] for v in files.values()),available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)))
''', 300))
    for value in files.values(): assert pin(ROOT / value['retained']) == value['identity']
    save(BASE / 'closed.json', dict(**result, snapshot=pin(BASE / 'snapshot.json'), generator=pin(Path(__file__)),
         scope='Four duplicate VM trace exports only. All local exports, archives, raw traces, clocks and immutable inputs retained.'))
    print(json.dumps(result))


if __name__ == '__main__': main()
