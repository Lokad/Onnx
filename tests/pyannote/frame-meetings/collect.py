"""Collect the new connected replay only after all owned process births terminate."""
from pathlib import Path
import json, subprocess, tarfile
from prepare import BASE,pin,read,write
from vm import ssh,KEY,HOST,REMOTE


def main():
    assert not (BASE/'collected').exists()
    script='''from pathlib import Path
import hashlib,json,sys,tarfile
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
base=Path(%r)
def pin(path):
 with path.open('rb') as stream:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())
births={}
for phase in ['inputs','run']:
 state=json.loads((base/('process-managed-'+phase)/'identity.json').read_text())
 assert (base/('process-managed-'+phase)/'complete.json').exists()
 for b in [state['supervisor'],state['child']]:
  if b:births[b['pid']]=b['birth']
 births.update({int(pid):birth for pid,birth in state['members'].items()})
deployment=json.loads((base/'deployment-managed.json').read_text());births[deployment['pid']]=deployment['birth']
for pid,birth in births.items():
 try:assert psutil.Process(pid).create_time()!=birth,('Live owned identity',pid,birth)
 except psutil.NoSuchProcess:pass
frozen=json.loads((base/'frozen.json').read_text())
for name,wanted in frozen['files'].items():assert pin(base/name)==wanted,name
manifest=json.loads((base/'manifest.json').read_text())
for item in manifest['models'].values():assert pin(base.parents[1]/item['amd_path'])=={k:item[k] for k in ['bytes','sha256']}
files={p.relative_to(base).as_posix():pin(p) for p in sorted(base.rglob('*')) if p.is_file()}
receipt=dict(terminal=True,births=[dict(pid=pid,birth=birth) for pid,birth in births.items()],files=files,frozen=pin(base/'frozen.json'))
with (base/'collection.json').open('x') as f:json.dump(receipt,f,indent=2)
archive=base.with_name(base.name+'-results.tar.gz');assert not archive.exists()
with tarfile.open(archive,'w:gz') as tar:
 for name in [*files,'collection.json']:tar.add(base/name,arcname=name,recursive=False)
print(json.dumps(dict(archive=pin(archive),receipt=pin(base/'collection.json'),files=len(files),births=receipt['births'])))
'''%REMOTE
    result=json.loads(ssh(script));archive=BASE/'results.tar.gz'
    assert not archive.exists()
    subprocess.run(['scp','-i',KEY,'-o','BatchMode=yes',HOST+':'+REMOTE+'-results.tar.gz',str(archive)],check=True)
    assert pin(archive)==result['archive'];target=BASE/'collected';target.mkdir()
    with tarfile.open(archive) as tar:tar.extractall(target,filter='data')
    assert pin(target/'collection.json')==result['receipt'];receipt=read(target/'collection.json')
    for name,wanted in receipt['files'].items():assert pin(target/name)==wanted,name
    assert {p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file()}==set(receipt['files'])|{'collection.json'}
    for name,wanted in read(BASE/'payload/preparation.json')['files'].items():assert pin(target/name)==wanted,name
    assert pin(target/'frozen.json')==read(BASE/'deployment.json')['frozen']
    write(BASE/'collection-transfer.json',result);print(json.dumps(result))


if __name__=='__main__':main()
