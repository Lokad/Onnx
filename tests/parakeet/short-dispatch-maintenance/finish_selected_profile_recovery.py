"""Finish local bookkeeping after the remote pre-worker recovery completed."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/parakeet/selected-profile-amd'))
from common import BASE,prepared,pin,read,save
from transport import PRELUDE,ssh


def main():
    spec=prepared();assert not (BASE/'preflight-recovery.json').exists()
    original=read(BASE/'deployment.json')
    assert original==read(BASE/'failed-preflight-observation.json')['owner']
    assert original['pid']==905229 and original['birth']==1790223933.75
    controller=Path(__file__).with_name('recover_selected_profile_preflight.py')
    receipt=json.loads(ssh(PRELUDE+f'''
sys.path.insert(0,str(base/'tools'));import remote
remote.idle();remote.verify()
value=read(base/'preflight-recovery.json')
assert value['passed'] and value['no_worker_started'] and value['inputs_unchanged']
assert value['original_owner']=={original!r} and not live(value['original_owner'])
assert value['controller']=={pin(controller)!r} and value['payload']=={spec['payload']!r}
assert not any((base/n).exists() for n in ['deployment.json','identity.json','logs','control','sampled-a','sampled-b'])
assert len(value['files'])==3
for name,wanted in value['files'].items():assert pin(base/name)==wanted,name
print(json.dumps(value))
''',300))
    source=BASE/'deployment.json';target=BASE/'failed-905229-deployment.json'
    assert source.resolve().is_relative_to(BASE.resolve()) and target.resolve().is_relative_to(BASE.resolve())
    assert not target.exists() and read(source)==receipt['original_owner']
    local_identity=pin(source);source.rename(target);assert pin(target)==local_identity
    save(BASE/'preflight-recovery.json',receipt)
    save(BASE/'local-recovery-completion.json',dict(passed=True,controller=pin(Path(__file__)),
        local_original=local_identity,remote_original=receipt['files'][target.name],identity_equal=True,
        reason='Original local/remote deployment JSON has identical parsed identity but different serialization; preserve both byte representations. Remote recovery had completed before the local byte-equality assertion failed.'))
    print(json.dumps(dict(passed=True,original_owner=original,local_original=local_identity,remote_original=receipt['files'][target.name])))


if __name__=='__main__':main()
