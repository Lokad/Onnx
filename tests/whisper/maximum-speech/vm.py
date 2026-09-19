"""Digest-verified deployment, read-only polling and one-time AMD collection."""
from pathlib import Path
import argparse
import json
import subprocess
import tarfile
from prepare import sha, read, write_new

KEY = 'C:/Users/JoannesVermorel/.ssh/id_onnx-bench.pem'
HOST = 'vermorel@74.178.91.76'
REMOTE = '/home/vermorel/Onnx/artifacts/whisper-maximum-speech-amd-20260919'


def ssh(script):
    result = subprocess.run(['ssh','-i',KEY,'-o','BatchMode=yes',HOST,'python3 -B -'],
        input=script,text=True,capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stdout+'\n'+result.stderr)
    return result.stdout


def extract(archive, destination):
    destination.mkdir()
    with tarfile.open(archive) as tar:
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and '\\' not in m.name for m in tar.getmembers())
        tar.extractall(destination,filter='data')


def deploy(base):
    assert not (base/'deployment-local.json').exists() and not (base/'deployment-output.txt').exists()
    pins = read(base/'preparation.json')
    assert sha(base/'payload.tar.gz') == pins['sha256']
    needed = pins['bytes']+pins['logical_bytes']-pins['borrowed_bytes']+8*1024**2
    preflight = json.loads(ssh('''from pathlib import Path
import json,os,subprocess
p=Path(REMOTE)
assert not p.exists() and not p.with_suffix('.tar.gz').exists()
assert subprocess.check_output(['git','-C','/home/vermorel/Onnx','rev-parse','HEAD'],text=True).strip()=='172181fc5ab4eb2bdc2eb7f37e80d25e482a0887'
for pid in Path('/proc').iterdir():
    if not pid.name.isdigit():continue
    try:
        raw=(pid/'cmdline').read_bytes().replace(b'\\x00',b' ')
        assert not raw.startswith(b'dotnet ') and b'RecordingReplay.dll' not in raw and b'Probe.dll' not in raw,(pid.name,raw)
    except (FileNotFoundError,ProcessLookupError):pass
v=os.statvfs(p.parent);free=v.f_bavail*v.f_frsize
assert free>NEEDED,free
print(json.dumps(dict(free_before=free)))
'''.replace('REMOTE',repr(REMOTE)).replace('NEEDED',str(needed))))
    subprocess.run(['scp','-i',KEY,'-o','BatchMode=yes',str(base/'payload.tar.gz'),HOST+':'+REMOTE+'.tar.gz'],check=True)
    script = '''from pathlib import Path
import hashlib,json,tarfile,subprocess,os
p=Path(REMOTE);archive=p.with_suffix('.tar.gz')
assert archive.stat().st_size==SIZE
with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==DIGEST
p.mkdir()
with tarfile.open(archive) as t:
    assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and '\\\\' not in m.name for m in t.getmembers())
    t.extractall(p,filter='data')
install=subprocess.run(['python3','-B',str(p/'remote.py'),'install',str(p)],text=True,capture_output=True)
assert install.returncode==0,install.stdout+install.stderr
launched=subprocess.run(['python3','-B',str(p/'remote.py'),'launch',str(p)],text=True,capture_output=True)
assert launched.returncode==0,launched.stdout+launched.stderr
v=os.statvfs(p)
print(json.dumps(dict(install=install.stdout,deployment=json.loads(launched.stdout),free_after=v.f_bavail*v.f_frsize)))
'''.replace('REMOTE',repr(REMOTE)).replace('SIZE',str(pins['bytes'])).replace('DIGEST',repr(pins['sha256']))
    # Preserve remote errors too: a partial deployment is never blindly retried.
    result = subprocess.run(['ssh','-i',KEY,'-o','BatchMode=yes',HOST,'python3 -B -'],input=script,text=True,capture_output=True)
    with (base/'deployment-output.txt').open('x',encoding='utf-8') as stream:
        stream.write(result.stdout+'\n'+result.stderr)
    assert result.returncode == 0,result.stderr
    record = dict(preflight=preflight,**json.loads(result.stdout))
    write_new(base/'deployment-local.json',record)
    print(json.dumps(record,indent=2))


def poll(base):
    print(ssh('''from pathlib import Path
import json,time
b=Path(REMOTE)
j=json.loads((b/'result/identity.json').read_text())
print(json.dumps(dict(complete=j['complete'],error=j.get('error'),elapsed=time.time()-j['started'],runs=j['runs']),indent=2))
for p in [j['supervisor']]+j['runs']:
    f=Path('/proc')/str(p['pid'])/'stat'
    if f.exists():
        parts=f.read_text().split(') ',1)[1].split()
        print('PROCESS',p['pid'],'expected',p['start'],'observed',parts[19],'state',parts[0])
    else:print('PROCESS',p['pid'],'absent')
print('OUTPUT',(b/'result/managed.stdout').read_text()[-1600:])
print('ERROR',(b/'result/managed.stderr').read_text()[-1000:])
'''.replace('REMOTE',repr(REMOTE))))


def collect(base):
    assert not (base/'download.json').exists() and not (base/'collected').exists() and not (base/'results.tar.gz').exists()
    record = json.loads(ssh('''import subprocess
r=subprocess.run(['python3','-B',REMOTE+'/remote.py','collect',REMOTE],text=True,capture_output=True)
assert r.returncode==0,r.stdout+r.stderr
print(r.stdout)
'''.replace('REMOTE',repr(REMOTE))))
    archive = base/'results.tar.gz'
    subprocess.run(['scp','-i',KEY,'-o','BatchMode=yes',HOST+':'+record['archive'],str(archive)],check=True)
    assert archive.stat().st_size == record['bytes'] and sha(archive) == record['sha256']
    extract(archive,base/'collected')
    assert sha(base/'collected/collection.json') == record['collection_sha256']
    write_new(base/'download.json',record)
    print(json.dumps(record,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('deploy','poll','collect'))
    parser.add_argument('--artifact',type=Path,required=True)
    args = parser.parse_args()
    globals()[args.action](args.artifact.resolve())
