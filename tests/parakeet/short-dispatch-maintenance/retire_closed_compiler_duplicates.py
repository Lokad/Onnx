"""Retire verified VM compiler-output duplicates only after Parakeet scoring."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tests/parakeet/packed-final-row-app-amd'))
from run import ssh, PRELUDE, pin, read, save, BASE as APP

BASE = ROOT / 'artifacts/e5-repeatability-compiler-retention-20260925'
CANDIDATES = ROOT / 'artifacts/e5-repeatability-retention-candidates-20260925.json'


def local():
    app = read(APP / 'closed.json')
    assert app['passed'] and app['analysis'] == pin(APP / 'analysis.json')
    receipt = read(APP / 'collected/collection.json')
    assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
    assert pin(APP / 'collected/collection.json') == app['files']['collected/collection.json']
    value = read(CANDIDATES)
    assert value['local_only'] and not value['removed'] and not value['remote_inspected']
    for row in value['candidates']:
        path = ROOT / row['local']
        assert path.is_relative_to(ROOT / 'artifacts') and path.name == 'instructions.json'
        folder = ROOT / Path(row['local']).parts[0] / Path(row['local']).parts[1]
        assert pin(folder / 'closed.json') == row['closure']
        proof = read(folder / 'closed.json')
        key = path.relative_to(folder).as_posix()
        if key not in proof['files']:
            key = path.relative_to(ROOT).as_posix()
        assert proof['passed'] and proof['files'][key] == pin(path) == row['identity']
        assert pin(folder / 'collected/collection.json') == row['collection']
        for name, wanted in row['sources'].items():
            assert pin(ROOT / name) == wanted
    return value['candidates'], receipt['identities']


COMMON = PRELUDE + '''
from protocol import pin,read
from remote import idle,live
idle();assert psutil.boot_time()==1789634288.0
protected=set();manifests={}
for folder in Path('/dev/shm').glob('lokad-*'):
 for name in ['payload.json','stage.json','spec.json']:
  path=folder/name
  if not path.is_file():continue
  record=read(path);manifests[str(path)]=pin(path)
  protected.update(str((folder/n).resolve()) for n in record.get('files',{}))
  protected.update(str(Path(n).resolve()) for n in record.get('external',{}))
  for link in record.get('links',{}).values():
   if isinstance(link,dict) and 'source' in link:protected.add(str(Path(link['source']).resolve()))
'''


def inspect():
    assert not BASE.exists()
    candidates, owners = local()
    result = json.loads(ssh(COMMON + f'''
rows={candidates!r};app_owners={owners!r}
assert not any(live(i) for i in app_owners)
selected={{}};skipped=[]
for row in rows:
 root=Path(row['root']);path=root/row['name']
 assert root.resolve()==root and root.parent==Path('/dev/shm')
 assert path.resolve()==path and path.is_relative_to(root) and path.name=='instructions.json'
 if not path.is_file():skipped.append(dict(path=str(path),reason='absent'));continue
 if str(path) in protected:skipped.append(dict(path=str(path),reason='protected'));continue
 if path.is_symlink() or path.stat().st_nlink!=1:skipped.append(dict(path=str(path),reason='linked'));continue
 if pin(root/'collection.json')!=row['collection']:
  skipped.append(dict(path=str(path),reason='different collection'));continue
 assert not any(live(i) for i in row['owners']) and pin(path)==row['identity']
 assert str(path) not in selected
 selected[str(path)]=dict(**row,physical_bytes=path.stat().st_blocks*512)
print(json.dumps(dict(passed=True,files=selected,skipped=skipped,manifests=manifests,app_owners=app_owners,
 before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free),
 physical_bytes=sum(r['physical_bytes'] for r in selected.values()))))
''', 300))
    BASE.mkdir()
    save(BASE / 'prospective.json', dict(**result, candidates=pin(CANDIDATES), generator=pin(Path(__file__))))
    print(json.dumps({k: v for k, v in result.items() if k not in ['files', 'manifests', 'app_owners']}
                     | dict(files=len(result['files']))))


def apply():
    assert not (BASE / 'closed.json').exists()
    local()
    value = read(BASE / 'prospective.json')
    assert value['passed'] and value['files'] and value['candidates'] == pin(CANDIDATES)
    assert value['generator'] == pin(Path(__file__))
    for row in value['files'].values():
        assert pin(ROOT / row['local']) == row['identity']
    result = json.loads(ssh(COMMON + f'''
value={value!r}
assert manifests==value['manifests'] and not any(live(i) for i in value['app_owners'])
for name,row in value['files'].items():
 root=Path(row['root']);path=Path(name)
 assert root.resolve()==root and root.parent==Path('/dev/shm')
 assert path==root/row['name'] and path.resolve()==path and path.is_relative_to(root)
 assert path.name=='instructions.json' and not path.is_symlink() and path.stat().st_nlink==1
 assert name not in protected and pin(path)==row['identity']
 assert pin(root/'collection.json')==row['collection'] and not any(live(i) for i in row['owners'])
 assert path.stat().st_blocks*512==row['physical_bytes']
before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
for name in value['files']:Path(name).unlink()
assert all(not Path(name).exists() for name in value['files'])
idle()
print(json.dumps(dict(passed=True,files=len(value['files']),physical_bytes=value['physical_bytes'],
 logical_bytes=sum(r['identity']['bytes'] for r in value['files'].values()),before=before,
 after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free))))
''', 300))
    for row in value['files'].values():
        assert pin(ROOT / row['local']) == row['identity']
    save(BASE / 'closed.json', dict(**result, prospective=pin(BASE / 'prospective.json'),
        all_local_originals_retained=True, scope='Only terminal compiler-output duplicates, after the scored Parakeet campaign.'))
    print(json.dumps(result))


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ['inspect', 'apply']
    globals()[sys.argv[1]]()
