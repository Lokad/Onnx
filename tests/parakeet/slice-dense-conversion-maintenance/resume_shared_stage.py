"""Complete the unstarted shared staging step without another transport or worker."""
import base64
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT/'tests/parakeet/slice-dense-conversion-shared-amd'
sys.path.insert(0, str(TOOLS))
from run import BASE, PRELUDE, ssh, prepared
from protocol import pin, read, save


def main():
    spec = prepared()
    receipt_dir = ROOT/'artifacts/parakeet-slice-dense-shared-stage-recovery-20260925'
    assert not receipt_dir.exists()
    assert (BASE/'stage-started.json').exists()
    assert not any((BASE/name).exists() for name in ['staged.json','payload.json','deployment.json','collected'])
    receipt_dir.mkdir()
    save(receipt_dir/'prospective.json', dict(
        helper=pin(Path(__file__)), prepared=pin(BASE/'prepared.json'),
        stage_started=read(BASE/'stage-started.json'),
        original_failure='run.py stage: unchanged remote_prepare.py line17 rejected available memory below11GiB after transfer/extraction; no payload, runtimes, deployment or worker was created.',
        idle_observation=dict(available=11867774976,tmpfs=4343730176,boot_time=1789634288.0),
        action='Verify every extracted byte, then invoke the unchanged staging main in the transport process. Preserve all limits and retain its log; no workload is launched.'))
    result = json.loads(ssh(PRELUDE+f'''
import base64, contextlib, io
from protocol import pin, read, save, verify, LIMITS
import remote, remote_prepare
remote.idle()
assert psutil.boot_time()==1789634288.0
assert not any((base/name).exists() for name in ['payload.json','staged.json','deployment.json','identity.json','runtimes','transfer.tar.gz','stage-recovery.json'])
assert pin(base/'stage.json')=={spec['stage']!r}
for name,wanted in read(base/'stage.json')['files'].items():
 assert pin(base/name)==wanted,name
before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage(base).free)
assert before['available']>=LIMITS['preflight_available'] and before['tmpfs']>=LIMITS['preflight_tmpfs']
save(base/'stage-recovery.json',dict(passed=False,before=before,no_worker=True,method='Unchanged remote_prepare.main in existing transport process'))
log=io.StringIO()
with contextlib.redirect_stdout(log):
 remote_prepare.main()
value=verify(base)
assert not remote.live(value['previous_owner'])
receipt=dict(passed=True,payload=pin(base/'payload.json'),files=len(value['files']),external=len(value['external']))
save(base/'staged.json',receipt)
print(json.dumps(dict(**receipt,before=before,log=log.getvalue(),payload_base64=base64.b64encode((base/'payload.json').read_bytes()).decode('ascii'))))
''', 300))
    encoded = result.pop('payload_base64')
    with (BASE/'payload.json').open('xb') as stream:
        stream.write(base64.b64decode(encoded))
    assert pin(BASE/'payload.json') == result['payload']
    save(BASE/'staged.json', {k: result[k] for k in ['passed','payload','files','external']})
    save(receipt_dir/'closed.json', dict(**result, prospective=pin(receipt_dir/'prospective.json')))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
