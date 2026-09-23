"""Recover only the local receipt of an already completed, unlaunched stage."""
import base64,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/parakeet/first-use-kernels-shared-amd'))
import run
from protocol import pin,read,save

def main():
    spec=run.prepared()
    assert not (run.BASE/'payload.json').exists() and not (run.BASE/'staged.json').exists()
    value=json.loads(run.ssh(run.PRELUDE+f'''
import base64
from protocol import pin,read,verify
from remote import idle,live
idle()
assert not (base/'deployment.json').exists() and not (base/'identity.json').exists()
assert pin(base/'stage.json')=={spec['stage']!r}
payload=verify(base);receipt=read(base/'staged.json')
assert receipt['passed'] and receipt['payload']==pin(base/'payload.json')
assert receipt['files']==len(payload['files']) and receipt['external']==len(payload['external'])
assert not live(payload['previous_owner'])
print(json.dumps(dict(receipt=receipt,payload=base64.b64encode((base/'payload.json').read_bytes()).decode('ascii'))))
''',300))
    receipt=value['receipt'];data=base64.b64decode(value['payload'],validate=True)
    (run.BASE/'payload.json').write_bytes(data);assert pin(run.BASE/'payload.json')==receipt['payload']
    save(run.BASE/'staged.json',receipt)
    save(run.BASE/'stage-receipt-recovery.json',dict(passed=True,reason='Stage controller exited1 without captured output; VM stage complete, local payload and receipt absent.',
        prepared=pin(run.BASE/'prepared.json'),script=pin(__file__),no_deployment_or_workers=True,remote_payload_unchanged=True,receipt=receipt))
    print(json.dumps(receipt))

if __name__=='__main__':main()
