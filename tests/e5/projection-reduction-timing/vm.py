"""Deploy once, poll exact Linux births, collect only after termination."""
import argparse, subprocess, tarfile
from common import *

HOST='vermorel@74.178.91.76';KEY='C:/Users/JoannesVermorel/.ssh/id_onnx-bench.pem'
REMOTE='/home/vermorel/Onnx/artifacts/e5-reduction-timing-20260920'


def ssh(script):
    return subprocess.check_output(['ssh','-i',KEY,'-o','BatchMode=yes',HOST,'python3 -B -'],input=script,text=True,encoding='utf-8')


def launch(base):
    assert not (base/'deployment.json').exists()
    wanted=pin(base/'payload.tar.gz')
    ssh('from pathlib import Path\nassert not Path(%r).exists() and not Path(%r).exists()' % (REMOTE,REMOTE+'.tar.gz'))
    subprocess.run(['scp','-i',KEY,str(base/'payload.tar.gz'),HOST+':'+REMOTE+'.tar.gz'],check=True)
    script='from pathlib import Path\nimport hashlib,json,tarfile,subprocess,os\nbase=Path(%r)\nwanted=%r\n' % (REMOTE,wanted)+'''
archive=base.with_name(base.name+'.tar.gz')
with archive.open('rb') as f:assert dict(bytes=archive.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())==wanted
assert not base.exists() and os.statvfs(base.parent).f_bavail*os.statvfs(base.parent).f_frsize>=128*1024**2
with tarfile.open(archive) as tar:
 members=tar.getmembers();assert len({m.name for m in members})==len(members)
 assert all(m.isfile() and not m.name.startswith('/') and '..' not in Path(m.name).parts for m in members)
 tar.extractall(base,filter='data')
subprocess.run(['python3','-B',str(base/'remote.py'),'launch',str(base)],check=True)
'''
    value=json.loads(ssh(script));assert value['bundle_sha256']==pin(base/'payload/bundle.json')['sha256'];write(base/'deployment.json',value);print(json.dumps(value))


def poll(base):
    script='from pathlib import Path\nimport sys,json\nbase=Path(%r)\nsys.path.insert(0,str(base))\nimport process_support as h\n' % REMOTE+'''
state=h.read(base/'result/identity.json');rows=[state['supervisor']]+[dict(pid=r['pid'],start=r['start']) for r in state['runs']]
print(json.dumps(dict(complete=state['complete'],code=state['code'],error=state.get('error'),runs=[dict(name=r['name'],code=r['code'],seconds=r.get('seconds')) for r in state['runs']],observed=[dict(expected=r,actual=h.proc(r['pid'])) for r in rows])))
'''
    print(ssh(script))


def collect(base):
    assert not (base/'download.json').exists()
    result=json.loads(ssh('import subprocess\nsubprocess.run(%r,check=True)' % ['python3','-B',REMOTE+'/remote.py','collect',REMOTE]))
    write(base/'download.json',result);archive=base/'results.tar.gz';assert not archive.exists()
    subprocess.run(['scp','-i',KEY,HOST+':'+result['archive'],str(archive)],check=True)
    assert pin(archive)=={k:result[k] for k in ['bytes','sha256']}
    with tarfile.open(archive) as tar:
        members=tar.getmembers();assert len({m.name for m in members})==len(members)
        assert all(m.isfile() and not m.name.startswith('/') and '..' not in Path(m.name).parts for m in members)
        tar.extractall(base/'collected',filter='data')
    receipt=read(base/'collected/collection.json');assert pin(base/'collected/collection.json')['sha256']==result['collection_sha256']
    assert {p.relative_to(base/'collected').as_posix() for p in (base/'collected').rglob('*') if p.is_file()}==set(receipt['files'])|{'collection.json'}
    for name,wanted in receipt['files'].items():assert pin(base/'collected'/name)==wanted,name
    print(json.dumps(dict(collected=len(receipt['files']),archive=pin(archive),complete=receipt['complete'],code=receipt['code'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','poll','collect']);p.add_argument('--artifact',type=Path,required=True);a=p.parse_args()
    globals()[a.action](a.artifact.resolve())
