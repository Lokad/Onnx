"""Collect completed AMD outputs; refuse live processes and preserve failed runs."""
from pathlib import Path, PurePosixPath, PureWindowsPath
import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from common import pin, read, write

HOST='vermorel@74.178.91.76'
KEY='C:/Users/JoannesVermorel/.ssh/id_onnx-bench.pem'
REMOTE='/home/vermorel/Onnx/artifacts/pyannote-natural-meetings-20260920'


def extract(archive,destination,expected):
    assert not destination.exists()
    with tarfile.open(archive,'r:gz') as stream:
        members=stream.getmembers();seen=set()
        for m in members:
            name=m.name
            assert m.isfile() and name not in seen and name in expected and m.size==expected[name]['bytes']
            assert '\\' not in name and ':' not in name and not PurePosixPath(name).is_absolute() and not PureWindowsPath(name).is_absolute()
            assert all(v not in ('','.','..') for v in name.split('/'))
            seen.add(name)
        assert seen==set(expected)
        destination.mkdir()
        for m in members:
            target=destination/m.name;assert target.resolve().is_relative_to(destination.resolve())
            target.parent.mkdir(parents=True,exist_ok=True)
            with stream.extractfile(m) as source,target.open('xb') as out:shutil.copyfileobj(source,out)
            assert pin(target)==expected[m.name]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);a=p.parse_args();base=a.artifact.resolve()
    assert base.name=='pyannote-natural-meetings-20260920' and not (base/'collected').exists()
    script=r'''
from pathlib import Path
import sys,json,hashlib,tarfile,time
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
base=Path('/home/vermorel/Onnx/artifacts/pyannote-natural-meetings-20260920')
def pin(p):
 with p.open('rb') as s:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(s,'sha256').hexdigest())
frozen=json.loads((base/'frozen.json').read_text())
for name,wanted in frozen['files'].items():assert pin(base/name)==wanted,name
manifest=json.loads((base/'manifest.json').read_text())
for item in manifest['models'].values():assert pin(base.parents[1]/item['amd_path'])=={k:item[k] for k in ['bytes','sha256']},item['amd_path']
births={}
for name in ['process-managed-inputs','process-managed-run']:
 state=json.loads((base/name/'identity.json').read_text());assert (base/name/'complete.json').exists()
 for item in [state['supervisor'],state['child']]:
  if item:births[item['pid']]=item['birth']
 births.update({int(pid):birth for pid,birth in state['members'].items()})
for pid,birth in births.items():
 try:assert psutil.Process(pid).create_time()!=birth,('Owned process still live',pid,birth)
 except psutil.NoSuchProcess:pass
receipt=base/'collection.json';archive=base.with_name(base.name+'-results.tar.gz')
if not receipt.exists():
 files={p.relative_to(base).as_posix():pin(p) for p in sorted(base.rglob('*')) if p.is_file() and p.relative_to(base).as_posix() not in frozen['files'] and p.name!='frozen.json'}
 value=dict(schema=1,all_owned_processes_terminal=True,terminal_processes=[dict(pid=pid,birth=birth) for pid,birth in births.items()],files=files,frozen=pin(base/'frozen.json'),verified_reusable_files=frozen['files'],created=time.time())
 with receipt.open('x') as s:json.dump(value,s,indent=2)
 with tarfile.open(archive,'x:gz') as s:
  for name in list(files)+['collection.json']:s.add(base/name,arcname=name,recursive=False)
value=json.loads(receipt.read_text())
for name,wanted in value['files'].items():assert pin(base/name)==wanted,name
print(json.dumps(dict(archive=pin(archive),receipt=pin(receipt),collection=value)))
'''
    result=subprocess.run(['ssh','-i',KEY,'-o','BatchMode=yes',HOST,'python3 -B -'],input=script,text=True,encoding='utf-8',capture_output=True,check=True)
    remote=json.loads(result.stdout)
    assert pin(base/'frozen.json')==remote['collection']['frozen']
    frozen=read(base/'frozen.json')
    assert remote['collection']['verified_reusable_files']==frozen['files']
    for name,wanted in frozen['files'].items():assert pin(base/name)==wanted,name
    archive=base/'amd-results.tar.gz'
    if not archive.exists():
        partial=base/'amd-results.partial';assert not partial.exists()
        subprocess.run(['scp','-i',KEY,HOST+':'+REMOTE+'-results.tar.gz',str(partial)],check=True)
        assert pin(partial)==remote['archive'];partial.rename(archive)
    assert pin(archive)==remote['archive']
    expected=dict(remote['collection']['files'],**{'collection.json':remote['receipt']})
    extract(archive,base/'collected',expected)
    assert read(base/'collected/collection.json')==remote['collection']
    # Install a verified local copy at the audit's canonical path, never overwriting a prior result.
    target=base/'process-managed-run';assert not target.exists()
    shutil.copytree(base/'collected/process-managed-run',target)
    for path in target.rglob('*'):
        if path.is_file():assert pin(path)==pin(base/'collected'/path.relative_to(base))
    write(base/'collection-check.json',dict(passed=True,remote=remote,files=len(expected),collector=pin(Path(__file__))))
    print('Collected',len(expected),'files, with all AMD births verified terminal.')


if __name__=='__main__':main()
