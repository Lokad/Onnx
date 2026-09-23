"""Resolve the recorded V1/V2 release path mismatch before the first M55 worker."""
import base64
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tests/parakeet/ordered-wide-blocks-build-amd'))
import run
from protocol import pin, read, save


def main():
    spec = run.prepared()
    incident = run.BASE/'staging-incident.json'
    value = read(incident)
    assert value['no_worker_launched'] and value['immutable_stage_verified']
    assert value['stage'] == spec['stage']
    closure = ROOT/'artifacts/parakeet-wide-entry-first-use-root-amd-v2-20260923/closed.json'
    assert read(closure)['passed'] and pin(closure) == value['parent_release']
    assert value['old_path'] == '/dev/shm/lokad-parakeet-wide-entry-first-use-root-20260923'
    assert value['corrected_path'] == '/dev/shm/lokad-parakeet-wide-entry-first-use-root-v2-20260923'
    for name in ['staged.json', 'payload.json', 'deployment.json', 'staging-recovery.json']:
        assert not (run.BASE/name).exists(), name
    result = json.loads(run.ssh(run.PRELUDE+f'''
import base64,contextlib,io
from protocol import read,pin,save,verify,LIMITS
import remote,remote_prepare
remote.idle()
assert psutil.boot_time()==1789634288.0
assert psutil.virtual_memory().available>=LIMITS['preflight_available']
assert psutil.disk_usage(base).free>=LIMITS['preflight_tmpfs']
for name in ['payload.json','identity.json','deployment.json','staged.json','measured','release-path-recovery.json','transfer.tar.gz']:
 assert not (base/name).exists(),name
assert pin(base/'stage.json')=={spec['stage']!r}
stage=read(base/'stage.json')
for name,wanted in stage['files'].items():assert pin(base/name)==wanted,name
assert stage['parent_release']=={value['parent_release']!r}
assert str(remote_prepare.RELEASE)=={value['old_path']!r}
corrected=Path({value['corrected_path']!r})
for row in {value['observations']!r}:
 assert pin(base/'evidence/release'/row['file'])==row['expected']
 assert pin(remote_prepare.RELEASE/row['file'])==row['incorrect']
 assert pin(corrected/row['file'])==row['corrected']==row['expected']
receipt=read(corrected/'collection.json')
assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
assert not any(remote.live(i) for i in receipt['identities'])
recovery=dict(passed=True,incident={pin(incident)!r},stage={spec['stage']!r},
 parent_release=stage['parent_release'],old_path=str(remote_prepare.RELEASE),corrected_path=str(corrected),
 frozen_remote_prepare=pin(base/'tools/remote_prepare.py'),resumer={pin(__file__)!r},
 no_worker_launched=True,source_and_protocol_unchanged=True)
save(base/'release-path-recovery.json',recovery)
remote_prepare.RELEASE=corrected
with contextlib.redirect_stdout(io.StringIO()):remote_prepare.main()
payload=verify(base)
assert not remote.live(payload['previous_owner'])
assert payload['files']['release-path-recovery.json']==pin(base/'release-path-recovery.json')
assert pin(base/'tools/remote_prepare.py')==recovery['frozen_remote_prepare']
staged=dict(passed=True,payload=pin(base/'payload.json'),files=len(payload['files']),external=len(payload['external']))
save(base/'staged.json',staged)
print(json.dumps(dict(**staged,recovery=recovery,
 payload_base64=base64.b64encode((base/'payload.json').read_bytes()).decode('ascii'))))
''', 300))
    encoded = result.pop('payload_base64')
    recovery = result.pop('recovery')
    (run.BASE/'payload.json').write_bytes(base64.b64decode(encoded))
    assert pin(run.BASE/'payload.json') == result['payload']
    save(run.BASE/'staged.json', result)
    save(run.BASE/'staging-recovery.json', dict(**recovery, staged=result))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
