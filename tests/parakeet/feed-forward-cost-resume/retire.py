"""Free only terminal VM trace duplicates after complete verified collection."""
import json
from pathlib import Path
import sys
from run import ROOT, OLD, OLD_REMOTE, original, paths, initial, prepared, pin, read, write, ssh


def main(mode):
    assert mode in ['original', 'stages', 'markers']
    initial()
    base, remote = (OLD, OLD_REMOTE) if mode == 'original' else paths(mode)
    if mode != 'original':
        prepared(mode)
        assert read(base / 'role-review.json')['passed']
    out = ROOT / f'artifacts/parakeet-feed-forward-cost-{mode}-trace-retention-20260925'
    assert not out.exists()
    folder = base / 'capture-collected'
    receipt = read(folder / 'capture-collection.json')
    assert receipt['terminal'] and receipt['code'] == (1 if mode == 'original' else 0)
    transfer = read(base / 'capture-transfer.json')
    assert transfer['passed'] and transfer['collection'] == pin(folder / 'capture-collection.json')
    assert transfer['archive'] == pin(base / 'capture-results.tar.gz')
    role = 'stages' if mode == 'original' else mode
    targets = {name:wanted for name,wanted in receipt['files'].items()
               if Path(name).parent.as_posix() == role and Path(name).name.startswith('cost-') and Path(name).suffix == '.json'}
    assert len(targets) == (33 if mode == 'original' else 80)
    for name,wanted in targets.items():
        assert pin(folder / name) == wanted
    prelude = original.PRELUDE.replace(OLD_REMOTE, remote)
    script = prelude + f'''
from remote import idle,live,pin,read,verify
idle();verify();assert psutil.boot_time()==1789634288.0
assert pin(base/'capture-collection.json')=={pin(folder/'capture-collection.json')!r}
owners={receipt['identities']!r};assert not any(live(i) for i in owners)
targets={targets!r}
for name,wanted in targets.items():
 path=base/name
 assert base.resolve()==base and base.parent==Path('/dev/shm')
 assert path.resolve()==path and path.is_relative_to(base) and path.parent==base/{role!r}
 assert not path.is_symlink() and path.stat().st_nlink==1 and pin(path)==wanted
'''
    prospective = ssh(script + '''
protected=set();manifests={}
for directory in Path('/dev/shm').glob('lokad-*'):
 for name in ['spec.json','stage.json','payload.json']:
  p=directory/name
  if not p.exists():continue
  value=read(p);manifests[str(p)]=pin(p)
  protected.update(str((directory/n).resolve()) for n in value.get('files',{}))
  protected.update(str(Path(n).resolve()) for n in value.get('external',{}))
assert not any(str(base/n) in protected for n in targets)
print(json.dumps(dict(passed=True,manifests=manifests,protected_paths=len(protected))))
''')
    assert prospective['passed']; out.mkdir()
    write(out / 'prospective.json', dict(**prospective, targets=targets, remote=remote,
        collection=pin(folder / 'capture-collection.json'), transfer=pin(base / 'capture-transfer.json'), helper=pin(__file__)))
    result = ssh(script + f'''
manifests={prospective['manifests']!r}
for name,wanted in manifests.items():assert pin(name)==wanted,name
assert set(manifests)=={{str(p) for d in Path('/dev/shm').glob('lokad-*') for n in ['spec.json','stage.json','payload.json'] if (p:=d/n).exists()}}
physical=sum((base/n).stat().st_blocks*512 for n in targets)
before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage(base).free)
for name in targets:(base/name).unlink()
assert all(not (base/n).exists() for n in targets)
print(json.dumps(dict(passed=True,files=len(targets),physical_bytes=physical,before=before,
 after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage(base).free))))
''')
    for name,wanted in targets.items():
        assert pin(folder / name) == wanted
    write(out / 'closed.json', dict(**result, prospective=pin(out / 'prospective.json'), all_local_originals_retained=True))
    print(json.dumps(result))


if __name__ == '__main__':
    main(sys.argv[1])
