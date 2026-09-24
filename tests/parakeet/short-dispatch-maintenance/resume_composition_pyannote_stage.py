"""Resume the unchanged Pyannote stage after its pre-mutation memory refusal."""
import base64
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tests/parakeet/validated-composition-pyannote-amd'))
import run
from protocol import pin, read, save


def main():
    spec = run.prepared()
    assert not any((run.BASE/name).exists() for name in ['staged.json','payload.json','deployment.json'])
    retirement = ROOT/'artifacts/parakeet-native-perf-remote-retention-20260924/closed.json'
    assert read(retirement)['passed']
    record = run.BASE/'stage-refusal-recovery.json'
    assert not record.exists()
    save(record, dict(passed=False, original_refusal='remote_prepare.py line 15: available memory/tmpfs preflight; before payload or runtime creation',
        original_stage_start=pin(run.BASE/'stage-started.json'), prepared=pin(run.BASE/'prepared.json'),
        recovery=pin(Path(__file__)), retirement=pin(retirement), inference_repeated=False))
    result = json.loads(run.ssh(run.PRELUDE + f'''
import base64
from protocol import pin,read,save,verify,LIMITS
from remote import idle,live
idle(); assert psutil.boot_time()==1789634288.0
assert not any((base/n).exists() for n in ['payload.json','runtimes','manifests','deployment.json','identity.json','staged.json'])
assert pin(base/'stage.json')=={spec['stage']!r}
stage=read(base/'stage.json')
assert {{p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file()}}==set(stage['files'])|{{'stage.json'}}
for name,wanted in stage['files'].items(): assert pin(base/name)==wanted,name
before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage(base).free)
assert before['available']>=LIMITS['preflight_available'] and before['tmpfs']>=LIMITS['preflight_tmpfs']
env=dict(os.environ,PYTHONPATH={run.SITE!r},PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONOPTIMIZE',None)
p=subprocess.run([sys.executable,'-B',str(base/'tools/remote_prepare.py')],cwd=base,env=env,text=True,capture_output=True,timeout=180)
assert p.returncode==0,(p.returncode,p.stdout[-3000:],p.stderr[-4000:])
value=verify(base);assert not live(value['previous_owner'])
receipt=dict(passed=True,payload=pin(base/'payload.json'),files=len(value['files']),external=len(value['external']))
save(base/'staged.json',receipt)
print(json.dumps(dict(**receipt,before=before,payload_base64=base64.b64encode((base/'payload.json').read_bytes()).decode('ascii'))))
''', 300))
    before = result.pop('before')
    encoded = result.pop('payload_base64')
    with (run.BASE/'payload.json').open('xb') as f: f.write(base64.b64decode(encoded))
    assert pin(run.BASE/'payload.json') == result['payload']
    save(run.BASE/'staged.json', result)
    value = read(record); value.update(passed=True, before=before, staged=pin(run.BASE/'staged.json'))
    save(record, value)
    print(json.dumps(result))


if __name__ == '__main__': main()
