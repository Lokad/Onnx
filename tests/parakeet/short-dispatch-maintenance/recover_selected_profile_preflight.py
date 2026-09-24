"""Preserve the proven pre-worker failure, then unblock the same frozen profile launch."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/parakeet/selected-profile-amd'))
from common import BASE,prepared,pin,read,save
from transport import PRELUDE,ssh


def main():
    spec=prepared();assert not (BASE/'preflight-recovery.json').exists()
    owner=read(BASE/'deployment.json');observation=read(BASE/'failed-preflight-observation.json')
    assert owner==observation['owner'] and owner['pid']==905229 and owner['birth']==1790223933.75
    maintenance=ROOT/'artifacts/parakeet-balanced-control-remote-retention-20260924/closed.json'
    assert read(maintenance)['passed'] and read(maintenance)['files']==8
    receipt=json.loads(ssh(PRELUDE+f'''
sys.path.insert(0,str(base/'tools'));import remote
remote.idle();remote.verify()
owner={owner!r}
assert read(base/'deployment.json')==owner and not live(owner)
assert pin(base/'payload.json')=={spec['payload']!r}
assert not any((base/n).exists() for n in ['identity.json','logs','control','sampled-a','sampled-b'])
assert (base/'supervisor.stdout').read_text()=={observation['stdout']!r}==''
assert (base/'supervisor.stderr').read_text()=={observation['stderr']!r}
assert "assert psutil.virtual_memory().available >= LIMITS['preflight_available']" in {observation['stderr']!r}
assert psutil.virtual_memory().available>=remote.LIMITS['preflight_available']
assert psutil.disk_usage(base).free>=remote.LIMITS['preflight_tmpfs']
files={{}}
for name in ['deployment.json','supervisor.stdout','supervisor.stderr']:
 source=base/name;target=base/('failed-905229-'+name)
 assert source.resolve().is_relative_to(base) and target.resolve().is_relative_to(base) and not target.exists()
 files[target.name]=pin(source);source.rename(target);assert pin(target)==files[target.name]
receipt=dict(passed=True,original_owner=owner,files=files,no_worker_started=True,inputs_unchanged=True,
 available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage(base).free,
 maintenance={pin(maintenance)!r},controller={pin(Path(__file__))!r},payload=pin(base/'payload.json'))
write(base/'preflight-recovery.json',receipt);print(json.dumps(receipt))
''',300))
    source=BASE/'deployment.json';target=BASE/'failed-905229-deployment.json'
    assert source.resolve().is_relative_to(BASE.resolve()) and target.resolve().is_relative_to(BASE.resolve())
    assert not target.exists() and pin(source)==receipt['files'][target.name]
    source.rename(target)
    save(BASE/'preflight-recovery.json',receipt)
    print(json.dumps(receipt))


if __name__=='__main__':main()
