"""Retire only closed VM profile duplicates; retain every exact local raw trace."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/parakeet/slice-dense-conversion-pyannote-amd'))
from run import ssh,PRELUDE
from protocol import pin,read,save
REMOTE_HELPERS="sys.path.insert(0,'/dev/shm/lokad-parakeet-slice-dense-conversion-shared-20260925/tools')\n"


def main():
    out=ROOT/'artifacts/parakeet-slice-dense-release-profile-retention-20260925'
    assert not out.exists()
    joint=ROOT/'artifacts/parakeet-slice-dense-conversion-profile-resume-amd-20260925'
    proof=read(joint/'closed.json')
    assert proof['passed'] and proof['analysis']==pin(joint/'analysis.json')
    assert pin(joint/'closed.json')['sha256']=='705fad8054d2f8020e99e80b32dca43f8f8c6ab046a63780a65faa062a427990'
    targets=[];sources=[];owners=[]
    for label,code,role in [('profile',1,'control'),('profile-resume',0,'wall')]:
        local=ROOT/f'artifacts/parakeet-slice-dense-conversion-{label}-amd-20260925/capture-collected'
        remote=f'/dev/shm/lokad-parakeet-slice-dense-conversion-{label}-20260925'
        receipt=read(local/'capture-collection.json');state=read(local/'capture-state.json')
        assert receipt['terminal'] and receipt['code']==state['code']==code and state['complete']
        assert receipt['state']==pin(local/'capture-state.json')
        owners += [state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
        names=sorted(n for n in receipt['files'] if n.startswith(role+'/phase-') and n.endswith('.json'))
        assert len(names)==80
        for name in names:
            assert pin(local/name)==receipt['files'][name]
            targets.append(dict(path=remote+'/'+name,local=(local/name).relative_to(ROOT).as_posix(),identity=receipt['files'][name]))
        sources.append(dict(root=remote,collection=pin(local/'capture-collection.json'),state=pin(local/'capture-state.json')))
    assert len(targets)==160
    script=PRELUDE+REMOTE_HELPERS+f'''
from protocol import pin,read
from remote import idle,live
idle();assert psutil.boot_time()==1789634288.0
owners={owners!r};targets={targets!r};sources={sources!r}
assert not any(live(i) for i in owners)
for s in sources:
 root=Path(s['root']);assert root.resolve()==root and root.parent==Path('/dev/shm')
 assert pin(root/'capture-collection.json')==s['collection'] and pin(root/'capture-state.json')==s['state']
protected=set();manifests={{}}
for folder in Path('/dev/shm').glob('lokad-*'):
 for name in ['payload.json','stage.json','spec.json']:
  path=folder/name
  if not path.exists():continue
  value=read(path);manifests[str(path)]=pin(path)
  protected.update(str((folder/n).resolve()) for n in value.get('files',{{}}))
  protected.update(str(Path(n).resolve()) for n in value.get('external',{{}}))
for row in targets:
 path=Path(row['path']);root=next(Path(s['root']) for s in sources if path.is_relative_to(Path(s['root'])))
 assert path.resolve()==path and path.is_relative_to(root) and not path.is_symlink()
 assert path.stat().st_nlink==1 and str(path) not in protected
 assert pin(path)==row['identity']
print(json.dumps(dict(passed=True,manifests=manifests,targets=targets,owners=owners,sources=sources,logical_bytes=sum(r['identity']['bytes'] for r in targets))))
'''
    prospective=json.loads(ssh(script));assert prospective['passed']
    out.mkdir();save(out/'prospective.json',dict(**prospective,helper=pin(Path(__file__)),joint_closure=pin(joint/'closed.json'),
        read_only_refusal='Initial inspector imported protocol from the not-yet-staged Pyannote namespace and failed before any mutation. Use the closed shared stage helpers; correct a draft unmatched parenthesis before execution.'))
    result=json.loads(ssh(PRELUDE+REMOTE_HELPERS+f'''
from protocol import pin,read
from remote import idle,live
idle();assert psutil.boot_time()==1789634288.0
targets={targets!r};owners={owners!r};sources={sources!r};manifests={prospective['manifests']!r}
assert not any(live(i) for i in owners)
for name,wanted in manifests.items():assert pin(name)==wanted,name
assert set(manifests)=={{str(p) for f in Path('/dev/shm').glob('lokad-*') for n in ['payload.json','stage.json','spec.json'] if (p:=f/n).exists()}}
before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
physical=0
for row in targets:
 path=Path(row['path']);assert path.resolve()==path and any(path.is_relative_to(Path(s['root'])) for s in sources)
 assert not path.is_symlink() and path.stat().st_nlink==1 and pin(path)==row['identity']
 physical+=path.stat().st_blocks*512
for row in targets:Path(row['path']).unlink()
assert all(not Path(row['path']).exists() for row in targets)
print(json.dumps(dict(passed=True,files=len(targets),logical_bytes=sum(r['identity']['bytes'] for r in targets),physical_bytes=physical,before=before,after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free))))
'''))
    for row in targets:assert pin(ROOT/row['local'])==row['identity']
    save(out/'closed.json',dict(**result,prospective=pin(out/'prospective.json'),all_local_originals_retained=True))
    print(json.dumps(result))


if __name__=='__main__':main()
