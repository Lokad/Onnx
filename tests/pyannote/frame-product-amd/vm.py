"""Single-use deployment and terminal-only collection of the declared payload."""
from pathlib import Path
import argparse, hashlib, json, subprocess, tarfile

KEY='C:/Users/JoannesVermorel/.ssh/id_onnx-bench.pem'
HOST='vermorel@74.178.91.76'


def pin(path):
    with path.open('rb') as stream:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def ssh(script):
    return subprocess.check_output(['ssh','-i',KEY,'-o','BatchMode=yes','-o','ConnectTimeout=15',HOST,'python3 -B -'],input=script,text=True,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['launch','collect']);parser.add_argument('artifact',type=Path);parser.add_argument('remote');args=parser.parse_args()
    base=args.artifact.resolve();remote=args.remote
    assert remote.startswith('/dev/shm/onnx-frame-product-') and '..' not in remote
    if args.action=='launch':
        preparation=json.loads((base/'preparation.json').read_text());assert pin(base/'payload.tar.gz')==preparation['archive']
        ssh("from pathlib import Path\nimport shutil\nassert shutil.disk_usage('/dev/shm').free>=2*1024**3\nPath(%r).mkdir()"%remote)
        subprocess.run(['scp','-i',KEY,'-o','BatchMode=yes',str(base/'payload.tar.gz'),HOST+':'+remote+'/payload.tar.gz'],check=True)
        script='''from pathlib import Path
import hashlib,json,tarfile,subprocess,sys,time
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
base=Path(%r)
with (base/'payload.tar.gz').open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==%r
with tarfile.open(base/'payload.tar.gz') as tar:tar.extractall(base,filter='data')
assert not (base/'deployment.json').exists() and not (base/'result').exists()
with (base/'supervisor.stdout').open('x') as out,(base/'supervisor.stderr').open('x') as err:
 child=subprocess.Popen(['python3','-B',str(base/'remote.py'),str(base)],stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True,cwd=base/'source')
 record=dict(pid=child.pid,birth=psutil.Process(child.pid).create_time(),started=time.time(),payload_sha256=%r)
 with (base/'deployment.json').open('x') as f:json.dump(record,f,indent=2)
 print(json.dumps(record))
'''%(remote,preparation['archive']['sha256'],preparation['payload']['sha256'])
        result=ssh(script)
        with (base/'deployment.json').open('x',encoding='utf-8') as stream:stream.write(result)
        print(result)
    else:
        assert not (base/'collected').exists() and not (base/'collection-transfer.json').exists()
        script='''from pathlib import Path
import hashlib,json,sys,tarfile
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
base=Path(%r);state=json.loads((base/'result/run.json').read_text());assert state['complete']
births=[state['supervisor']]
for run in state['runs']:births.extend(dict(pid=int(pid),birth=birth) for pid,birth in run['members'].items())
for identity in births:
 try:assert psutil.Process(identity['pid']).create_time()!=identity['birth'],identity
 except psutil.NoSuchProcess:pass
def pin(p):
 with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
files={p.relative_to(base).as_posix():pin(p) for p in sorted(base.rglob('*')) if p.is_file()}
with (base/'collection.json').open('x') as f:json.dump(dict(terminal=True,births=births,files=files),f,indent=2)
target=Path(str(base)+'-collected.tar.gz');assert not target.exists()
with tarfile.open(target,'w:gz') as tar:
 for p in sorted(base.rglob('*')):
  if p.is_file():tar.add(p,arcname=p.relative_to(base).as_posix(),recursive=False)
print(json.dumps(dict(archive=pin(target),files=len(files),remote=str(target),births=births)))
'''%remote
        transfer=json.loads(ssh(script));archive=base/'collected.tar.gz'
        subprocess.run(['scp','-i',KEY,'-o','BatchMode=yes',HOST+':'+transfer['remote'],str(archive)],check=True)
        assert pin(archive)==transfer['archive']
        target=base/'collected';target.mkdir()
        with tarfile.open(archive) as tar:tar.extractall(target,filter='data')
        receipt=json.loads((target/'collection.json').read_text())
        for name,value in receipt['files'].items():assert pin(target/name)==value,name
        with (base/'collection-transfer.json').open('x',encoding='utf-8') as stream:json.dump(transfer,stream,indent=2)
        print(json.dumps(dict(archive=transfer['archive'],files=len(receipt['files']),births=len(transfer['births']))))


if __name__=='__main__':main()
