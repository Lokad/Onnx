"""Collect the completed Linux reference, excluding its installed package tree."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'natural-meetings'))
from common import pin,read,write
from collect import extract,HOST,KEY

REMOTE='/home/vermorel/Onnx/artifacts/asr-native-linux-20260920'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact',type=Path,required=True)
    p.add_argument('--original',type=Path,required=True)
    a=p.parse_args();base=a.artifact.resolve();original=a.original.resolve()
    target=original/'native-linux-collected'
    assert base.name=='asr-native-linux-20260920' and not target.exists()
    script=r'''
from pathlib import Path
import sys,json,hashlib,tarfile,time
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
base=Path('/home/vermorel/Onnx/artifacts/asr-native-linux-20260920')
def pin(p):
 with p.open('rb') as s:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(s,'sha256').hexdigest())
def read(p):return json.loads(p.read_text())
frozen=read(base/'frozen.json');manifest=read(base/'manifest.json')
assert frozen['native_files']==manifest['native_files']
for name,wanted in frozen['files'].items():assert pin(base/name)==wanted,name
for name,wanted in frozen['native_files'].items():assert pin(Path(name))==wanted,name
for family in manifest['families'].values():
 for item in family['models'].values():assert pin(base.parents[1]/item['path'])=={k:item[k] for k in ['bytes','sha256']},item['path']
for case in manifest['cases']:
 item=case['audio'];assert pin(base.parents[1]/item['path'])=={k:item[k] for k in ['bytes','sha256']}
assert read(base/'campaign-native.json')==dict(complete=True,outcomes=[dict(family='whisper',code=0)])
births={r['pid']:r['birth'] for r in frozen['input_smoke']['terminal_processes']}
deployment=read(base/'deployment-native.json');births[deployment['pid']]=deployment['birth']
for mode in ['inputs','run']:
 folder=base/('process-native-whisper-'+mode);state=read(folder/'identity.json')
 assert state['complete'] is True and state['code']==0 and read(folder/'complete.json')==dict(code=0)
 for item in [state['supervisor'],state['child']]:births[item['pid']]=item['birth']
 births.update({int(pid):birth for pid,birth in state['members'].items()})
for pid,birth in births.items():
 try:assert psutil.Process(pid).create_time()!=birth,('Owned process still live',pid,birth)
 except psutil.NoSuchProcess:pass
receipt=base/'collection.json';archive=base.with_name(base.name+'-results.tar.gz')
if not receipt.exists():
 files={}
 for top in sorted(base.iterdir()):
  if top.name=='python':continue
  assert not top.is_symlink(),top
  for path in ([top] if top.is_file() else sorted(top.rglob('*'))):
   assert not path.is_symlink(),path
   if path.is_file():files[path.relative_to(base).as_posix()]=pin(path)
 value=dict(schema=1,all_owned_processes_terminal=True,terminal_processes=[dict(pid=p,birth=b) for p,b in sorted(births.items())],files=files,
  frozen=pin(base/'frozen.json'),verified_native_files=frozen['native_files'],excluded_directory='python',created=time.time())
 with receipt.open('x') as s:json.dump(value,s,indent=2)
 with tarfile.open(archive,'x:gz') as s:
  for name in list(files)+['collection.json']:s.add(base/name,arcname=name,recursive=False)
value=read(receipt)
for name,wanted in value['files'].items():assert pin(base/name)==wanted,name
assert value['verified_native_files']==frozen['native_files']
print(json.dumps(dict(archive=pin(archive),receipt=pin(receipt),collection=value)))
'''
    result=subprocess.run(['ssh','-i',KEY,'-o','BatchMode=yes',HOST,'python3 -B -'],input=script,text=True,encoding='utf-8',capture_output=True,check=True)
    remote=json.loads(result.stdout);launch=read(base/'launch-check.json')
    assert remote['collection']['frozen']==launch['frozen']
    archive=base/'linux-results.tar.gz';assert not archive.exists()
    partial=base/'linux-results.partial';assert not partial.exists()
    subprocess.run(['scp','-i',KEY,HOST+':'+REMOTE+'-results.tar.gz',str(partial)],check=True)
    assert pin(partial)==remote['archive'];partial.rename(archive)
    extract(archive,target,dict(remote['collection']['files'],**{'collection.json':remote['receipt']}))
    assert read(target/'frozen.json')==launch['frozen_value'] and read(target/'manifest.json')==launch['manifest']
    # Keep the Linux preparation failures and local structural proofs inside the final receipt.
    local=original/'native-linux-local';local.mkdir()
    for path in sorted(base.iterdir()):
        assert path.is_file(),path
        shutil.copyfile(path,local/path.name);assert pin(local/path.name)==pin(path)
    write(original/'native-linux-collection-check.json',dict(passed=True,remote=remote,collector=pin(Path(__file__)),files=len(remote['collection']['files'])+1))
    print('Collected Linux reference:',len(remote['collection']['files'])+1,'files; all owned births terminal.')


if __name__=='__main__':main()
