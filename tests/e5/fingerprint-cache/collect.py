"""Collect a terminal fingerprint experiment with exact archive and process identities."""
from pathlib import Path
import argparse,importlib.util,json,subprocess,sys
from generate import pin

HOST='vermorel@74.178.91.76'
KEY='C:/Users/JoannesVermorel/.ssh/id_onnx-bench.pem'
REMOTE='/home/vermorel/Onnx/artifacts/e5-fingerprint-cache-v3-20260920'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);a=p.parse_args();base=a.artifact.resolve()
    assert not (base/'collected').exists()
    script=r'''
from pathlib import Path
import sys,json,hashlib,tarfile,time
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
base=Path('/home/vermorel/Onnx/artifacts/e5-fingerprint-cache-v3-20260920')
def pin(p):
 with p.open('rb') as s:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(s,'sha256').hexdigest())
frozen=json.loads((base/'frozen.json').read_text())
for name,wanted in frozen['files'].items():assert pin(base/name)==wanted,name
assert pin(base.parents[1]/'models/multilingual-e5-small/model.onnx')==frozen['model']
state=json.loads((base/'timing-process/identity.json').read_text());assert state['complete'] is True and state['code']==0
deployment=json.loads((base/'deployment.json').read_text())
assert state['supervisor']=={k:deployment[k] for k in ['pid','birth']}
births={deployment['pid']:deployment['birth']}
assert len(state['runs'])==4
for r in state['runs']:
 assert r['code']==0
 births.update({int(p):b for p,b in r['members'].items()})
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
 frozen=pin(base/'frozen.json'),verified_model=frozen['model'],files=files)
with receipt.open('x') as s:json.dump(value,s,indent=2)
with tarfile.open(archive,'x:gz') as s:
 for name in list(files)+['collection.json']:s.add(base/name,arcname=name,recursive=False)
print(json.dumps(dict(collection=value,receipt=pin(receipt),archive=pin(archive))))
'''
    result=subprocess.run(['ssh','-i',KEY,'-o','BatchMode=yes',HOST,'python3 -B -'],input=script,text=True,encoding='utf-8',capture_output=True,check=True)
    remote=json.loads(result.stdout);assert remote['collection']['frozen']==pin(base/'deployment/frozen.json')
    archive=base/'results.tar.gz';partial=base/'results.partial';assert not archive.exists() and not partial.exists()
    subprocess.run(['scp','-i',KEY,HOST+':'+REMOTE+'-results.tar.gz',str(partial)],check=True)
    assert pin(partial)==remote['archive'];partial.rename(archive)
    original=Path(__file__).resolve().parents[2]/'audio/natural-meetings'
    sys.path.insert(0,str(original))
    spec=importlib.util.spec_from_file_location('safe_natural_collection',original/'collect.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.extract(archive,base/'collected',dict(remote['collection']['files'],**{'collection.json':remote['receipt']}))
    with (base/'collection-check.json').open('x') as s:json.dump(dict(passed=True,remote=remote,collector=pin(Path(__file__)),extractor=pin(original/'collect.py')),s,indent=2)
    print('Collected',len(remote['collection']['files'])+1,'files; all original worker and supervisor births terminal.')


if __name__=='__main__':main()
