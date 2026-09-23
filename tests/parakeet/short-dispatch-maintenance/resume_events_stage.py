"""Resume unchanged M42 preparation after its pre-worker memory preflight failure."""
import base64,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'tests/parakeet/dispatch-events-amd'))
from run import BASE,PRELUDE,SITE,ssh,prepared
from protocol import read,pin,save
spec=prepared();assert read(BASE/'extracted.json')['passed'] and not (BASE/'staged.json').exists()
result=json.loads(ssh(PRELUDE+f'''
import base64
from protocol import pin,save,verify
import remote
remote.idle()
assert not any((base/n).exists() for n in ['payload.json','deployment.json','identity.json','staged.json'])
assert pin(base/'stage.json')=={spec['stage']!r}
env=dict(os.environ,PYTHONPATH={SITE!r},PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONOPTIMIZE',None)
p=subprocess.run([sys.executable,'-B',str(base/'tools/remote_prepare.py')],cwd=base,env=env,text=True,capture_output=True,timeout=180)
assert p.returncode==0,(p.returncode,p.stdout[-3000:],p.stderr[-4000:])
value=verify(base);assert not remote.live(value['previous_owner'])
receipt=dict(passed=True,payload=pin(base/'payload.json'),files=len(value['files']),external=len(value['external']))
save(base/'staged.json',receipt)
print(json.dumps(dict(**receipt,payload_base64=base64.b64encode((base/'payload.json').read_bytes()).decode('ascii'))))
''',300))
encoded=result.pop('payload_base64');(BASE/'payload.json').write_bytes(base64.b64decode(encoded));assert pin(BASE/'payload.json')==result['payload']
save(BASE/'staged.json',result);print(json.dumps(result))
