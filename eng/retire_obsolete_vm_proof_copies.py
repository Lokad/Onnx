"""Retire exact local duplicates of old VM proof snapshots after all work is idle."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT/'eng/retire-collected-vm-duplicates-20260927.py'
loader = importlib.util.spec_from_file_location('verified_vm_retirement', ENGINE)
retirement = importlib.util.module_from_spec(loader)
loader.loader.exec_module(retirement)
retirement.OUT = ROOT/'artifacts/vm-retention-obsolete-proof-copies-20260927'
OUT = retirement.OUT
pin, read, save = retirement.pin, retirement.read, retirement.save

retirement.REMOTE_COMMON += r'''
# Protect every declared external file and link source, including staged specs.
for folder in Path('/dev/shm').glob('lokad-*'):
 for kind in ['payload.json','stage.json','spec.json']:
  path=folder/kind
  if not path.is_file():continue
  value=json.loads(path.read_text());payloads[str(path)]=pin(path)
  protected.update(os.path.normpath(str(folder/name)) for name in value.get('files',{}))
  external.update(os.path.normpath(name) for name in value.get('external',{}))
  for link in value.get('links',{}).values():
   if isinstance(link,dict) and isinstance(link.get('source'),str) and Path(link['source']).is_absolute():
    external.add(str(Path(link['source']).resolve()))
protected.update(external)

retained_roots={
 '/dev/shm/lokad-parakeet-decoder-packed-row-root-20260927',
 '/dev/shm/lokad-parakeet-decoder-packed-row-pyannote-app-20260927',
 '/dev/shm/lokad-parakeet-observed-dense-where-pyannote-app-20260924',
 '/dev/shm/lokad-pyannote-blocked-spatial-app-20260922'}

def eligible(folder,name):
 if folder.parent!=Path('/dev/shm') or folder.resolve()!=folder:return False
 if str(folder) in retained_roots or folder.name.startswith('lokad-lstmlayout'):return False
 relative=Path(name)
 if relative.is_absolute() or '..' in relative.parts or not name.startswith('evidence/'):return False
 if relative.suffix not in {'.json','.jsonl','.csv','.trx'}:return False
 p=folder/relative
 if str(p) in external or not p.is_file() or p.is_symlink() or p.resolve()!=p:return False
 # Retire only this terminal stage's private proof copy. All source, models,
 # assets, runtimes, input metadata and current LSTM namespaces remain intact.
 return p.stat().st_nlink==1 and p.stat().st_size>=4096
'''


def configuration():
    return dict(driver=pin(Path(__file__)), engine=pin(ENGINE),
        remote_guard=pin(OUT/'remote-common.py'),
        scope='Only old evidence/ files retained exactly locally; current LSTM namespaces, predecessor roots, every declared external input and all products remain.')


def verify_configuration():
    assert read(OUT/'configuration.json') == configuration()
    assert (OUT/'remote-common.py').read_text() == retirement.REMOTE_COMMON


def inventory():
    retirement.inventory()  # Includes a live-process check before VM inspection.
    with (OUT/'remote-common.py').open('x') as stream:
        stream.write(retirement.REMOTE_COMMON)
    save(OUT/'configuration.json', configuration())


def plan():
    verify_configuration()
    assert not (OUT/'plan.json').exists() and not (OUT/'supplement.json').exists()
    value = read(OUT/'inventory.json')
    selected, excluded, allocated = [], [], 0
    for row in value['candidates']:
        folder = ROOT/value['roots'][row['root']]['local']
        source = (folder/'collected'/row['name']).resolve()
        assert source.is_relative_to((ROOT/'artifacts').resolve())
        actual = pin(source) if source.is_file() else None
        if actual != row['wanted']:
            excluded.append(dict(root=row['root'], name=row['name'], actual=actual,
                reason='Exact local copy unavailable; keep VM file'))
            continue
        selected.append(dict(row, local=source.relative_to(ROOT).as_posix()))
        allocated += row['allocated']
        if allocated >= 2*1024**3:
            break
    assert selected, 'No verified obsolete proof copies to retire'
    assert len({(r['root'],r['name']) for r in selected}) == len(selected)
    assert len({(r['device'],r['inode']) for r in selected}) == len(selected)
    save(OUT/'supplement.json', dict(payloads=value['payloads'], candidates=[]))
    save(OUT/'plan.json', dict(source=pin(ENGINE), inventory=pin(OUT/'inventory.json'),
        supplement=pin(OUT/'supplement.json'), configuration=pin(OUT/'configuration.json'),
        retired_files=selected, expected_allocated_reclaimed=allocated, excluded=excluded,
        preservation=configuration()['scope']))
    print(json.dumps(dict(files=len(selected), expected_allocated_reclaimed=allocated,
        roots=len({r['root'] for r in selected}), excluded=len(excluded))))


def retire():
    verify_configuration()
    assert read(OUT/'plan.json')['configuration'] == pin(OUT/'configuration.json')
    retirement.retire()  # Rechecks idle, terminal owners, metadata, file IDs and every byte.
    save(OUT/'closed.json', dict(passed=True, retired=pin(OUT/'retired.json'),
        plan=pin(OUT/'plan.json'), configuration=pin(OUT/'configuration.json')))


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ['inventory','plan','retire']
    globals()[sys.argv[1]]()
