"""Finish verified staging after its memory preflight stopped before any worker."""
import base64,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/parakeet/short-dispatch-numerics'))
import run
from protocol import read,pin,save
spec=run.prepared()
assert not (run.BASE/'staged.json').exists() and not (run.BASE/'deployment.json').exists()
maintenance=ROOT/'artifacts/parakeet-short-dispatch-maintenance-20260923/closed.json'
assert read(maintenance)['passed']
result=json.loads(run.ssh(run.PRELUDE+f"""
import base64
from protocol import read,pin,save,verify
import remote
remote.idle()
assert not (base/'payload.json').exists() and not (base/'identity.json').exists() and not (base/'staged.json').exists()
assert not (base/'transfer.tar.gz').exists()
assert pin(base/'stage.json')=={spec['stage']!r}
for name,wanted in read(base/'stage.json')['files'].items():assert pin(base/name)==wanted,name
links=read(base/'fixture-links.json');assert links['passed'] and len(links['files'])==45
for row in links['files']:
 assert pin(base/'fixtures'/row['file'])==row['identity']==pin(Path(links['prior'])/'fixtures'/row['file'])
env=dict(os.environ,PYTHONPATH={run.SITE!r},PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONOPTIMIZE',None)
p=subprocess.run([sys.executable,'-B',str(base/'tools/remote_prepare.py')],cwd=base,env=env,text=True,capture_output=True,timeout=180)
assert p.returncode==0,(p.returncode,p.stdout[-3000:],p.stderr[-4000:])
value=verify(base);assert not remote.live(value['previous_owner'])
receipt=dict(passed=True,payload=pin(base/'payload.json'),files=len(value['files']),external=len(value['external']))
save(base/'staged.json',receipt)
print(json.dumps(dict(**receipt,payload_base64=base64.b64encode((base/'payload.json').read_bytes()).decode('ascii'))))
""",300))
encoded=result.pop('payload_base64');(run.BASE/'payload.json').write_bytes(base64.b64decode(encoded));assert pin(run.BASE/'payload.json')==result['payload']
save(run.BASE/'staged.json',result)
save(run.BASE/'staging-recovery.json',dict(passed=True,reason='First remote_prepare stopped at available-memory preflight before payload or workers; resume same verified stage after archived build-cache retirement.',maintenance=pin(maintenance),inputs_unchanged=True,staged=result))
print(json.dumps(result))
