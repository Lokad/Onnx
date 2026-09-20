"""Collect the immutable deployed payload and terminal AMD correctness results."""
from pathlib import Path
import argparse,hashlib,json,subprocess,tarfile

HOST='vermorel@74.178.91.76'
KEY='C:/Users/JoannesVermorel/.ssh/id_onnx-bench.pem'

def pin(path):
    with path.open('rb') as stream:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);a=p.parse_args();base=a.artifact.resolve()
    assert base.name.startswith('e5-fingerprint-product-') and all(c.isalnum() or c=='-' for c in base.name)
    remote_path='/home/vermorel/Onnx/artifacts/'+base.name
    assert not (base/'collected').exists()
    script=r'''
from pathlib import Path
import sys,json,hashlib,tarfile,time
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
base=Path('REMOTE_PATH')
def pin(p):
 with p.open('rb') as s:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(s,'sha256').hexdigest())
meta=json.loads((base/'frozen.json').read_text())
for name,wanted in meta['files'].items():assert pin(base/name)==wanted,name
for name,wanted in meta['assets'].items():assert pin(base.parents[1]/name)==wanted,name
state=json.loads((base/'result-amd/identity.json').read_text())
deployment=json.loads((base/'deployment.json').read_text());assert state['supervisor']=={k:deployment[k] for k in ['pid','birth']}
births={deployment['pid']:deployment['birth']}
for run in state['runs']:births.update({int(p):b for p,b in run['members'].items()})
for pid,birth in births.items():
 try:assert psutil.Process(pid).create_time()!=birth,('Owned process live',pid,birth)
 except psutil.NoSuchProcess:pass
receipt=base/'collection.json';archive=base.with_name(base.name+'-results.tar.gz')
assert not receipt.exists() and not archive.exists()
files={}
for path in sorted(base.rglob('*')):
 assert not path.is_symlink(),path
 if path.is_file():files[path.relative_to(base).as_posix()]=pin(path)
value=dict(schema=1,created=time.time(),all_owned_processes_terminal=True,births=[dict(pid=p,birth=b) for p,b in sorted(births.items())],
 frozen=pin(base/'frozen.json'),files=files)
with receipt.open('x') as stream:json.dump(value,stream,indent=2)
with tarfile.open(archive,'x:gz') as stream:
 for name in list(files)+['collection.json']:stream.add(base/name,arcname=name,recursive=False)
print(json.dumps(dict(collection=value,receipt=pin(receipt),archive=pin(archive))))
'''.replace('REMOTE_PATH',remote_path)
    response=subprocess.run(['ssh','-i',KEY,'-o','BatchMode=yes',HOST,'python3 -B -'],input=script,text=True,encoding='utf-8',capture_output=True,check=True)
    remote=json.loads(response.stdout);assert remote['collection']['frozen']==pin(base/'payload/frozen.json')
    archive=base/'results.tar.gz';assert not archive.exists()
    subprocess.run(['scp','-i',KEY,HOST+':'+remote_path+'-results.tar.gz',str(archive)],check=True);assert pin(archive)==remote['archive']
    expected=remote['collection']['files']|{'collection.json':remote['receipt']}
    with tarfile.open(archive) as tar:
        names=[m.name for m in tar.getmembers()];assert len(names)==len(set(names)) and set(names)==set(expected)
        assert all(m.isfile() and not m.name.startswith('/') and '..' not in Path(m.name).parts for m in tar.getmembers())
        tar.extractall(base/'collected',filter='data')
    for name,wanted in expected.items():assert pin(base/'collected'/name)==wanted,name
    with (base/'collection-check.json').open('x') as stream:json.dump(dict(passed=True,remote=remote),stream,indent=2)
    print('Collected',len(expected),'files; every observed process birth is terminal.')

if __name__=='__main__':main()
