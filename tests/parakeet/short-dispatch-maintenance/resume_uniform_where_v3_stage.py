"""Resume the unchanged V3 numerical campaign after its pre-worker memory check refused staging."""
import base64
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tests/parakeet/scalar-where-numerics-v3'))
import run
from protocol import pin, read, save


def main():
    spec = run.prepared()
    maintenance = ROOT/'artifacts/parakeet-scalar-where-v3-build-retention-20260924/closed.json'
    assert read(maintenance)['passed']
    for name in ['payload.json','staged.json','deployment.json','staging-recovery.json']:
        assert not (run.BASE/name).exists(), name
    result = json.loads(run.ssh(run.PRELUDE+f'''
import base64
from protocol import read,pin,save,verify,LIMITS
import remote
remote.idle()
assert psutil.boot_time()==1789634288.0
assert psutil.virtual_memory().available>=LIMITS['build_preflight_available']
assert psutil.disk_usage(base).free>=LIMITS['preflight_tmpfs']
for name in ['payload.json','identity.json','deployment.json','staged.json','fixtures','transfer.tar.gz','staging-recovery.json']:
 assert not (base/name).exists(),name
assert pin(base/'stage.json')=={spec['stage']!r}
for name,wanted in read(base/'stage.json')['files'].items():
 if not name.startswith('fixtures/'):assert pin(base/name)==wanted,name
recovery=dict(passed=True,reason='Original remote preparation stopped at the first available-memory check, before fixture links, payload or any worker.',
 maintenance={pin(maintenance)!r},stage={spec['stage']!r},resumer={pin(__file__)!r},
 original_source_tools_protocol_and_limits_unchanged=True)
save(base/'staging-recovery.json',recovery)
env=dict(os.environ,PYTHONPATH={run.SITE!r},PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONOPTIMIZE',None)
p=subprocess.run([sys.executable,'-B',str(base/'tools/remote_prepare.py')],cwd=base,env=env,text=True,capture_output=True,timeout=180)
assert p.returncode==0,(p.returncode,p.stdout[-3000:],p.stderr[-4000:])
payload=verify(base);assert not remote.live(payload['previous_owner'])
assert payload['files']['staging-recovery.json']==pin(base/'staging-recovery.json')
receipt=dict(passed=True,payload=pin(base/'payload.json'),files=len(payload['files']),external=len(payload['external']))
save(base/'staged.json',receipt)
print(json.dumps(dict(**receipt,recovery=recovery,payload_base64=base64.b64encode((base/'payload.json').read_bytes()).decode('ascii'))))
''',300))
    encoded = result.pop('payload_base64')
    recovery = result.pop('recovery')
    (run.BASE/'payload.json').write_bytes(base64.b64decode(encoded))
    assert pin(run.BASE/'payload.json') == result['payload']
    save(run.BASE/'staged.json',result)
    save(run.BASE/'staging-recovery.json',dict(**recovery,staged=result))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
