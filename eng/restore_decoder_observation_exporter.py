"""Restore exact retired exporter copies and finish only the pending preparation."""
import base64
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tests/parakeet/decoder-projection-observation'))
import run
from protocol import pin, read, save

OUT = ROOT/'artifacts/parakeet-decoder-projection-exporter-recovery-20260927'
FAILED = ROOT/'artifacts/parakeet-decoder-projection-stage-recovery-20260927'
EXPORT = ROOT/'artifacts/parakeet-dispatch-full-export-amd-20260923'
OLD = 'assert target.is_relative_to(BASE.resolve()) and not target.exists()'
NEW = "assert target.is_relative_to(BASE.resolve())\n        if target.exists():\n            assert pin(target) == link['identity']\n            continue"


def main():
    assert not OUT.exists(), 'Preserve the one-time restore attempt'
    spec = run.prepared()
    assert not (run.BASE/'staged.json').exists() and not (run.BASE/'payload.json').exists()
    failed = read(FAILED/'failed.json')
    assert not failed['passed'] and not failed['campaign_started']
    for name, wanted in failed['files'].items(): assert pin(FAILED/name) == wanted
    state = read(FAILED/'failure-state.json')
    stage = read(run.BASE/'bundle/stage.json')
    missing = state['missing_sources']
    assert len(missing) == 14 and len(state['existing_targets']) == 55
    assert set(stage['links']) == set(missing) | set(state['existing_targets'])
    assert pin(EXPORT/'closed.json')['sha256'] == '35b18e874e1e0c47a7b9e1fe6f2608b0940c1eb431c289c32ba05855cb2b5afa'
    proof = read(EXPORT/'closed.json'); assert proof['passed']
    for name, link in missing.items():
        assert name.startswith('export-runtime/') and link == stage['links'][name]
        assert pin(EXPORT/'collected'/name) == link['identity'] == proof['files']['collected/'+name]
    original = (run.BASE/'bundle/tools/remote_prepare.py').read_text()
    assert original.count(OLD) == 1
    resumed = original.replace(OLD, NEW)
    compile(resumed, '<remaining-preparation>', 'exec')
    OUT.mkdir()
    (OUT/'remote-prepare-resume.py').write_text(resumed, encoding='utf8')
    with tarfile.open(OUT/'exporter.tar.gz', 'w:gz') as archive:
        for name in sorted(missing): archive.add(EXPORT/'collected'/name, arcname=name, recursive=False)
    save(OUT/'intent.json', dict(passed=True, campaign_started=False,
        previous_failure=pin(FAILED/'failed.json'), prepared=pin(run.BASE/'prepared.json'),
        tool=pin(Path(__file__)), original_preparation=pin(run.BASE/'bundle/tools/remote_prepare.py'),
        resumed_preparation=pin(OUT/'remote-prepare-resume.py'), exporter_qualification=pin(EXPORT/'closed.json'),
        restored={n:v['identity'] for n,v in missing.items()}, archive=pin(OUT/'exporter.tar.gz'),
        scope='All existing targets are verified; restore only fourteen exact qualified local files. The original preparation differs only by accepting preverified existing targets. No diagnostic, resource, worker, product or scoring change.'))
    preflight = json.loads(run.ssh(run.PRELUDE+f'''
from protocol import pin,read,LIMITS
import remote
remote.idle()
assert psutil.boot_time()==1789634288.0
assert psutil.virtual_memory().available>=LIMITS['preflight_available']+256*1024**2
assert psutil.disk_usage('/dev/shm').free>=LIMITS['preflight_tmpfs']+256*1024**2
assert not any((base/name).exists() for name in ['payload.json','deployment.json','identity.json','staged.json','restore-exporter.tar.gz','stage-resume.stdout','stage-resume.stderr'])
assert pin(base/'stage.json')=={spec['stage']!r}
stage=read(base/'stage.json')
for name,wanted in stage['files'].items():assert pin(base/name)==wanted,name
for name,wanted in {state['existing_targets']!r}.items():
 assert pin(base/name)==wanted==stage['links'][name]['identity']
 assert pin(Path(stage['links'][name]['source']))==wanted
for name in {list(missing)!r}:assert not (base/name).exists()
assert pin(base/'stage-prepare.stderr')=={state['stderr_identity']!r}
assert pin(base/'stage-prepare.stdout')=={state['stdout_identity']!r}
print(json.dumps(dict(passed=True,existing_targets=55,absent_targets=14,available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)))
''', 60))
    save(OUT/'preflight.json', preflight)
    subprocess.run(['scp', *run.SSH[1:-1], str(OUT/'exporter.tar.gz'),
                    run.SSH[-1]+':'+run.REMOTE+'/restore-exporter.tar.gz'],
                   check=True, timeout=180, creationflags=subprocess.CREATE_NO_WINDOW)
    script = run.PRELUDE+f'''
import base64,contextlib,io,signal,traceback
from protocol import pin,read,save,verify
import remote
remote.idle()
assert not any((base/name).exists() for name in ['payload.json','deployment.json','identity.json','staged.json'])
archive=base/'restore-exporter.tar.gz'
assert pin(archive)=={pin(OUT/'exporter.tar.gz')!r}
with tarfile.open(archive) as tar:
 members=tar.getmembers()
 assert len(members)==14 and {{m.name for m in members}}==set({list(missing)!r})
 assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and not (base/m.name).exists() for m in members)
 tar.extractall(base,filter='data')
stage=read(base/'stage.json')
for name,link in stage['links'].items():assert pin(base/name)==link['identity'],name
archive.unlink()
save(base/'exporter-restoration.json',{read(OUT/'intent.json')!r})
source=(base/'tools/remote_prepare.py').read_text()
assert pin(base/'tools/remote_prepare.py')=={pin(run.BASE/'bundle/tools/remote_prepare.py')!r}
assert source.count({OLD!r})==1
source=source.replace({OLD!r},{NEW!r})
with (base/'stage-resume-source.py').open('x') as stream:stream.write(source)
assert pin(base/'stage-resume-source.py')=={pin(OUT/'remote-prepare-resume.py')!r}
out,err=io.StringIO(),io.StringIO()
def deadline(signum,frame):raise TimeoutError('Preparation exceeded180seconds')
signal.signal(signal.SIGALRM,deadline);signal.alarm(180)
try:
 with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
  namespace=dict(__name__='verified_preparation_resume',__file__=str(base/'tools/remote_prepare.py'))
  exec(compile(source,str(base/'tools/remote_prepare.py'),'exec'),namespace)
  namespace['main']()
except BaseException:
 err.write(traceback.format_exc());raise
finally:
 signal.alarm(0)
 with (base/'stage-resume.stdout').open('x') as stream:stream.write(out.getvalue())
 with (base/'stage-resume.stderr').open('x') as stream:stream.write(err.getvalue())
value=verify(base);assert not remote.live(value['previous_owner'])
receipt=dict(passed=True,payload=pin(base/'payload.json'),files=len(value['files']),external=len(value['external']))
save(base/'staged.json',receipt)
print(json.dumps(dict(**receipt,payload_base64=base64.b64encode((base/'payload.json').read_bytes()).decode('ascii'))))
'''
    compile(script, '<exporter-restoration>', 'exec')
    (OUT/'remote.py').write_text(script, encoding='utf8')
    raw = run.ssh(script, 240)
    (OUT/'remote.stdout').write_text(raw, encoding='utf8')
    value = json.loads(raw)
    encoded = value.pop('payload_base64')
    (run.BASE/'payload.json').write_bytes(base64.b64decode(encoded))
    assert pin(run.BASE/'payload.json') == value['payload']
    save(run.BASE/'staged.json', value)
    save(OUT/'closed.json', dict(passed=True, campaign_started=False,
        files={p.name:pin(p) for p in OUT.iterdir() if p.is_file()},
        payload=pin(run.BASE/'payload.json'), staged=pin(run.BASE/'staged.json')))
    print(json.dumps(dict(passed=True, recovery=pin(OUT/'closed.json'), staged=value)))


if __name__ == '__main__': main()
