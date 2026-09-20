"""Collect a terminal AMD comparison, including failures and every raw sample."""
import json, subprocess, tarfile
from pathlib import Path
from deploy import BASE,REMOTE,KEY,HOST,ssh
from protocol import pin,read,write


def main():
    assert not (BASE/'collected').exists()
    script='''from pathlib import Path
import sys,json,hashlib,tarfile
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
base=Path(%r);state=json.loads((base/'campaign/identity.json').read_text());assert state['complete'] is True
def pin(p):
 with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
births={state['supervisor']['pid']:state['supervisor']['birth']}
deployment=json.loads((base/'deployment.json').read_text());births[deployment['pid']]=deployment['birth']
for run in state['runs']:
 assert run['complete'] is True
 births.update({int(pid):birth for pid,birth in run['members'].items()})
for pid,birth in births.items():
 try:assert psutil.Process(pid).create_time()!=birth,('live',pid,birth)
 except psutil.NoSuchProcess:pass
frozen=json.loads((base/'frozen.json').read_text())
for name,wanted in frozen['files'].items():assert pin(base/name)==wanted,name
for name,wanted in frozen['external'].items():assert pin(Path(name))==wanted,name
files={p.relative_to(base).as_posix():pin(p) for p in sorted(base.rglob('*')) if p.is_file()}
receipt=dict(terminal=True,code=state['code'],files=files,births=[dict(pid=p,birth=b) for p,b in births.items()],frozen=pin(base/'frozen.json'),external_verified=len(frozen['external']))
with (base/'collection.json').open('x') as f:json.dump(receipt,f,indent=2)
archive=Path('/dev/shm/audio-amd-comparison-v2-20260920-results.tar.gz');assert not archive.exists()
with tarfile.open(archive,'w:gz') as tar:
 for name in [*files,'collection.json']:tar.add(base/name,arcname=name,recursive=False)
print(json.dumps(dict(archive_path=str(archive),archive=pin(archive),receipt=pin(base/'collection.json'),files=len(files),births=receipt['births'])))
'''%REMOTE
    result=json.loads(ssh(script));archive=BASE/'results.tar.gz';assert not archive.exists()
    subprocess.run(['scp','-i',KEY,'-o','BatchMode=yes',HOST+':'+result['archive_path'],str(archive)],check=True)
    assert pin(archive)==result['archive'];target=BASE/'collected';target.mkdir()
    with tarfile.open(archive) as tar:tar.extractall(target,filter='data')
    assert pin(target/'collection.json')==result['receipt'];receipt=read(target/'collection.json')
    for name,wanted in receipt['files'].items():assert pin(target/name)==wanted,name
    assert {p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file()}==set(receipt['files'])|{'collection.json'}
    assert pin(target/'frozen.json')==pin(BASE/'frozen.json')==read(BASE/'deployment.json')['frozen']
    write(BASE/'collection-transfer.json',result);print(json.dumps(result))


if __name__=='__main__':main()
