"""Retire closed VM tensor duplicates while keeping all exact local raw results."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/parakeet/slice-dense-conversion-pyannote-amd'))
from run import ssh,PRELUDE
from protocol import pin,read,save

PINS={'models':'5860fc366c3d976907f35f8d7a1832c10566fa08910af32c168dca7a63979842',
      'shared':'2f96025addac77c4fe336e55929659e6827d71e0d9ea6412fc65a7fe88d64047',
      'pyannote':'10237093de2f33c996c5d919e09d9278e553357f5b23ef30aa8046d2cb25b836'}


def main():
    out=ROOT/'artifacts/parakeet-slice-dense-release-array-retention-20260925'
    assert not out.exists()
    targets=[];sources=[];owners=[]
    for label,digest in PINS.items():
        base=ROOT/f'artifacts/parakeet-slice-dense-conversion-{label}-amd-20260925'
        root=f'/dev/shm/lokad-parakeet-slice-dense-conversion-{label}-20260925'
        proof=read(base/'closed.json');receipt=read(base/'collected/collection.json')
        assert pin(base/'closed.json')['sha256']==digest and proof['passed']
        assert proof['analysis']==pin(base/'analysis.json')
        assert proof['files']['collected/collection.json']==pin(base/'collected/collection.json')
        assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
        owners.extend(receipt['identities']);jobs=read(base/'payload.json')['jobs']
        names=sorted(n for n in receipt['files'] if any(n.startswith(j+'/') for j in jobs) and Path(n).suffix in ['.bin','.f32'])
        assert len(names)==dict(models=3136,shared=332,pyannote=48)[label]
        for name in names:
            path=base/'collected'/name;wanted=receipt['files'][name]
            assert pin(path)==wanted==proof['files']['collected/'+name]
            targets.append(dict(path=root+'/'+name,local=path.relative_to(ROOT).as_posix(),identity=wanted))
        sources.append(dict(root=root,collection=pin(base/'collected/collection.json'),closure=pin(base/'closed.json')))
    assert len(targets)==3516 and sum(r['identity']['bytes'] for r in targets)==118547624
    common=PRELUDE+f'''
from protocol import pin,read
from remote import idle,live
idle();assert psutil.boot_time()==1789634288.0
targets={targets!r};sources={sources!r};owners={owners!r}
assert not any(live(i) for i in owners)
for row in sources:
 root=Path(row['root']);assert root.resolve()==root and root.parent==Path('/dev/shm')
 assert pin(root/'collection.json')==row['collection']
for row in targets:
 path=Path(row['path']);assert path.resolve()==path and any(path.is_relative_to(Path(s['root'])) for s in sources)
 assert not path.is_symlink() and path.stat().st_nlink==1 and pin(path)==row['identity']
'''
    prospective=json.loads(ssh(common+'''
paths=set();manifests={}
for folder in Path('/dev/shm').glob('lokad-*'):
 for name in ['payload.json','stage.json','spec.json']:
  path=folder/name
  if not path.exists():continue
  value=read(path);manifests[str(path)]=pin(path)
  paths.update(str(folder/n) for n in value.get('files',{}))
  paths.update(value.get('external',{}))
protected={str(Path(n).resolve()) for n in paths}
assert not any(row['path'] in protected for row in targets)
print(json.dumps(dict(passed=True,manifests=manifests,protected_paths=len(protected))))
''',300))
    assert prospective['passed'];out.mkdir()
    save(out/'prospective.json',dict(**prospective,targets=targets,sources=sources,owners=owners,helper=pin(Path(__file__)),logical_bytes=118547624))
    result=json.loads(ssh(common+f'''
manifests={prospective['manifests']!r}
for name,wanted in manifests.items():assert pin(name)==wanted,name
assert set(manifests)=={{str(p) for f in Path('/dev/shm').glob('lokad-*') for n in ['payload.json','stage.json','spec.json'] if (p:=f/n).exists()}}
before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
physical=sum(Path(r['path']).stat().st_blocks*512 for r in targets)
for row in targets:Path(row['path']).unlink()
assert all(not Path(r['path']).exists() for r in targets)
print(json.dumps(dict(passed=True,files=len(targets),logical_bytes=118547624,physical_bytes=physical,before=before,after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free))))
''',300))
    for row in targets:assert pin(ROOT/row['local'])==row['identity']
    save(out/'closed.json',dict(**result,prospective=pin(out/'prospective.json'),all_local_originals_retained=True))
    print(json.dumps(result))


if __name__=='__main__':main()
