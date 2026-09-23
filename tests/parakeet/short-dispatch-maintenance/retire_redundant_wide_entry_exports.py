"""Remove only remote duplicate inventories/events whose complete proofs remain local."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tests/parakeet/isolated-short-kernels-build-amd-v2'))
from run import ssh, PRELUDE
from protocol import pin, read, save

BASE = ROOT / 'artifacts/parakeet-redundant-wide-entry-export-retention-20260923'
TARGETS = [
    ('parakeet-wide-projection-isolation-build-amd-20260923', 'lokad-parakeet-wide-projection-isolation-build-20260923',
     'c67b21b2e3f9d3d09f822ab075e649e232c6c42d7e7503470711d21a1bf5604d', ['inventory/instructions.json']),
    ('parakeet-wide-entry-first-use-build-amd-20260923', 'lokad-parakeet-wide-entry-first-use-build-20260923',
     'da923692f2c97cbff2774006f4a3dc911aaca39e0af5b35a89639443a75ead58', ['inventory/instructions.json']),
    ('parakeet-isolated-runtime-diagnostic-amd-20260923', 'lokad-parakeet-isolated-runtime-diagnostic-20260923',
     'ecfea393d0475617f32de3b87419cd6354f6d3b61edd4883adbdba877caa0f2c',
     ['candidate-export/events/events.jsonl', 'current-export/events/events.jsonl']),
    ('parakeet-wide-runtime-diagnostic-amd-20260923', 'lokad-parakeet-wide-runtime-diagnostic-20260923',
     'fee8b6d9a20917ca24a03300f35720d644b7841ba2ad5408f71392f11bf7642c',
     ['candidate-export/events/events.jsonl', 'current-export/events/events.jsonl']),
]


def main():
    assert not BASE.exists()
    roots = {}; files = {}
    for local, remote, digest, names in TARGETS:
        folder = ROOT / 'artifacts' / local
        assert pin(folder / 'closed.json')['sha256'] == digest
        closure = read(folder / 'closed.json'); assert closure['passed']
        for name, wanted in closure['files'].items(): assert pin(folder / name) == wanted, name
        roots['/dev/shm/' + remote] = pin(folder / 'collected/collection.json')
        for name in names:
            kept = folder / 'collected' / name
            assert closure['files']['collected/' + name] == pin(kept)
            files['/dev/shm/' + remote + '/' + name] = dict(identity=pin(kept), retained=kept.relative_to(ROOT).as_posix())
    BASE.mkdir()
    preflight = PRELUDE + f'''
from protocol import read,pin
from remote import live,idle
idle(); assert psutil.boot_time()==1789634288.0
roots={roots!r}; files={files!r}
protected=set()
for folder in Path('/dev/shm').glob('lokad-*'):
 if not (folder/'payload.json').exists():continue
 spec=read(folder/'payload.json')
 protected.update(str((folder/n).resolve()) for n in spec.get('files',{{}}))
 protected.update(str(Path(n).resolve()) for n in spec.get('external',{{}}))
for name,wanted in roots.items():
 root=Path(name); assert root.resolve().parent==Path('/dev/shm')
 assert pin(root/'collection.json')==wanted
 receipt=read(root/'collection.json')
 assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
 assert not any(live(i) for i in receipt['identities'])
 spec=read(root/'payload.json')
 for n,w in spec['files'].items(): assert pin(root/n)==w,n
 for n,w in spec['external'].items(): assert pin(Path(n))==w,n
for name,value in files.items():
 path=Path(name)
 assert not path.is_symlink() and str(path.resolve())==name and name not in protected
 assert any(path.is_relative_to(Path(r)) for r in roots)
 assert pin(path)==value['identity']
'''
    snapshot = json.loads(ssh(preflight + "print(json.dumps(dict(passed=True,available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)))", 300))
    save(BASE / 'snapshot.json', dict(**snapshot, roots=roots, files=files))
    result = json.loads(ssh(preflight + '''
for name in files: Path(name).unlink()
assert all(not Path(n).exists() for n in files)
print(json.dumps(dict(passed=True,files=len(files),bytes=sum(v['identity']['bytes'] for v in files.values()),available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)))
''', 300))
    for value in files.values(): assert pin(ROOT / value['retained']) == value['identity']
    save(BASE / 'closed.json', dict(**result, snapshot=pin(BASE / 'snapshot.json'), generator=pin(Path(__file__)),
        scope='Six remote duplicate exports only. Complete local copies and all raw traces, inputs and products retained.'))
    print(json.dumps(result))


if __name__ == '__main__': main()
