"""Collect complete raw evidence after all recorded process identities are terminal."""
import json,subprocess,tarfile
from common import BASE,REMOTE,KEY,HOST,PRELUDE,pin,read,write,ssh


def main():
    assert not (BASE/'collected').exists()
    script=PRELUDE+'''state=read(base/'run/identity.json');assert state['complete']
births=[state['supervisor']]+[dict(pid=int(p),birth=b) for p,b in state['members'].items()]
deployment=read(base/'deployment.json');assert deployment['pid']==state['supervisor']['pid'] and deployment['birth']==state['supervisor']['birth']
terminal(births);frozen=read(base/'frozen.json')
for name,wanted in frozen['files'].items():assert pin(base/name)==wanted,name
for name,wanted in frozen['external'].items():assert pin(name)==wanted,name
files={p.relative_to(base).as_posix():pin(p) for p in sorted(base.rglob('*')) if p.is_file()}
receipt=dict(terminal=True,code=state['code'],files=files,births=births,frozen=pin(base/'frozen.json'),external_verified=len(frozen['external']))
write(base/'collection.json',receipt)
archive=Path('/dev/shm/whisper-memory-contracts-amd-20260921-results.tar.gz');assert not archive.exists()
with tarfile.open(archive,'w:gz') as tar:
 for name in [*files,'collection.json']:tar.add(base/name,arcname=name,recursive=False)
print(json.dumps(dict(archive_path=str(archive),archive=pin(archive),receipt=pin(base/'collection.json'),files=len(files),births=births)))
'''
    value=json.loads(ssh(script));archive=BASE/'results.tar.gz';assert not archive.exists()
    subprocess.run(['scp','-i',KEY,'-o','BatchMode=yes',HOST+':'+value['archive_path'],str(archive)],check=True)
    assert pin(archive)==value['archive'];target=BASE/'collected';target.mkdir()
    with tarfile.open(archive) as tar:tar.extractall(target,filter='data')
    receipt=read(target/'collection.json');assert pin(target/'collection.json')==value['receipt']
    for name,wanted in receipt['files'].items():assert pin(target/name)==wanted,name
    assert {p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file()}==set(receipt['files'])|{'collection.json'}
    assert pin(target/'frozen.json')==pin(BASE/'frozen.json')==read(BASE/'deployment.json')['frozen']
    write(BASE/'collection-transfer.json',value);print(json.dumps(value))


if __name__=='__main__':main()
