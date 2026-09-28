"""Restore the original in-directory layout of staged application inputs, preserving bytes."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tests/parakeet/transpose-axis-ort-profile-amd'))
import run

OUT = ROOT/'artifacts/vm-transpose-profile-input-restoration-20260928'
APP = ROOT/'artifacts/parakeet-transpose-axis-app-amd-20260928'
OFFLOAD = ROOT/'artifacts/vm-transpose-closed-evidence-offload-20260928'
STORE = '/home/vermorel/Onnx/artifacts/transpose-closed-evidence-20260928'
pin, read, save = run.pin, run.read, run.write


def main():
    assert not OUT.exists()
    failure = run.BASE/'failed.json'
    assert pin(failure)['sha256'] == 'e2b318d499c3081497cd198b3a9899f60461e9efe5fc4cd1b88af87c856e62d5'
    assert read(failure)['no_inference_executed']
    proof = read(APP/'closed.json'); assert proof['passed'] and proof['admitted']
    assert pin(APP/'collected/collection.json') == proof['files']['collected/collection.json']
    assert pin(APP/'payload.json') == proof['files']['payload.json']
    offload = read(OFFLOAD/'plan.json'); completed = read(OFFLOAD/'completed.json')
    assert completed['passed']
    allowed = {r['path']:r for r in offload['files'] if r['path'].startswith(run.REMOTE_APP+'/')}
    common = run.transport.PRELUDE + f'''
import shutil
app=Path({run.REMOTE_APP!r});store=Path({STORE!r})
sys.path.insert(0,str(app/'tools'))
from protocol import pin,read,verify
from remote import idle,live
idle();assert psutil.boot_time()==1789634288.0
assert not live({read(failure)['owner']!r})
assert app.resolve()==app and app.parent==Path('/dev/shm')
assert read(store/'completed.json')=={completed!r}
assert pin(app/'payload.json')=={pin(APP/'payload.json')!r}
assert pin(app/'collection.json')=={pin(APP/'collected/collection.json')!r}
receipt=read(app/'collection.json')
assert receipt['terminal'] and receipt['code']==0 and not any(live(i) for i in receipt['identities'])
payload=read(app/'payload.json');allowed={allowed!r}
rows=[]
for name,wanted in payload['files'].items():
 path=app/name;resolved=path.resolve()
 assert pin(path)==wanted
 if resolved.is_relative_to(app):continue
 assert path.is_symlink() and str(path) in allowed
 previous=allowed[str(path)]
 assert resolved==Path(previous['target']) and resolved.is_relative_to(store)
 assert wanted==previous['identity'] and not resolved.is_symlink()
 rows.append(dict(path=str(path),source=str(resolved),identity=wanted))
assert rows and sum(r['identity']['bytes'] for r in rows)<64*1024**2
'''
    prospective = run.transport.ssh(common + '''
print(json.dumps(dict(passed=True,rows=rows,available=psutil.virtual_memory().available,
 tmpfs=psutil.disk_usage('/dev/shm').free)))
''')
    OUT.mkdir()
    save(OUT/'prepared.json', dict(**prospective, helper=pin(Path(__file__)),
         failure=pin(failure), original_application=pin(APP/'closed.json'),
         offload=pin(OFFLOAD/'completed.json')))
    result = run.transport.ssh(common + f'''
assert rows=={prospective['rows']!r}
before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
for row in rows:
 path=Path(row['path']);source=Path(row['source']);temp=path.with_name(path.name+'.restored-input')
 assert not temp.exists() and not temp.is_symlink() and temp.parent.resolve().is_relative_to(app)
 assert path.is_symlink() and path.resolve()==source and pin(source)==row['identity']
 with source.open('rb') as src,temp.open('xb') as dst:
  shutil.copyfileobj(src,dst);dst.flush();os.fsync(dst.fileno())
 shutil.copystat(source,temp)
 assert pin(temp)==row['identity']
 os.replace(temp,path)
 assert path.resolve()==path and not path.is_symlink() and pin(path)==pin(source)==row['identity']
verify(app)
print(json.dumps(dict(passed=True,files=len(rows),bytes=sum(r['identity']['bytes'] for r in rows),
 original_verifier_passed=True,all_bytes_preserved=True,disk_backups_retained=True,before=before,
 after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free))))
''')
    save(OUT/'closed.json', dict(**result, preparation=pin(OUT/'prepared.json')))
    print(json.dumps(result))


if __name__ == '__main__': main()
