"""Freeze a new timing scope using already passed, unchanged conformance evidence."""
from pathlib import Path
import json, subprocess, sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/audio/amd-comparison'))
from deploy import ssh,KEY,HOST,REMOTE as PRIOR_REMOTE,BASE as PRIOR_BASE
from protocol import pin,read,write

BASE=ROOT/'artifacts/audio-amd-two-family-20260920'
REMOTE='/home/vermorel/Onnx/artifacts/audio-amd-two-family-20260920'


def main():
    assert not BASE.exists();BASE.mkdir()
    prior=PRIOR_BASE/'collected';closed=read(PRIOR_BASE/'failure-closed.json')
    assert closed['closure_passed'] and not closed['campaign_passed']
    for name,wanted in closed['files'].items():assert pin(ROOT/name)==wanted,name
    receipt=read(prior/'collection.json');state=read(prior/'campaign/identity.json')
    assert state['code']==1 and state['complete'] and len(state['runs'])==6
    workers=state['runs'][:4]
    assert [(r['family'],r['engine']) for r in workers]==[('parakeet','ort'),('parakeet','managed'),('pyannote','ort'),('pyannote','managed')]
    assert all(r['complete'] and r['code']==0 and 'error' not in r for r in workers)
    gate=dict(passed=True,scope=['parakeet','pyannote'],prior_artifact=PRIOR_REMOTE,prior_frozen=pin(prior/'frozen.json'),
              prior_collection=pin(prior/'collection.json'),failure_closure=pin(PRIOR_BASE/'failure-closed.json'),workers=workers,calls=48)
    external={PRIOR_REMOTE+'/'+name:wanted for name,wanted in receipt['files'].items() if any(name.startswith(r['output']+'/') for r in workers)}
    external[PRIOR_REMOTE+'/collection.json']=gate['prior_collection'];external[PRIOR_REMOTE+'/frozen.json']=gate['prior_frozen']
    for name,wanted in external.items():assert pin(prior/name.removeprefix(PRIOR_REMOTE+'/'))==wanted,name
    write(BASE/'prior-gate.json',gate)
    folder=Path(__file__).resolve().parent;source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    script='''from pathlib import Path
import hashlib,json,os,sys
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
old=Path(%r);base=Path(%r)
def pin(p):
 with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
frozen=json.loads((old/'frozen.json').read_text())
assert pin(old/'frozen.json')==%r
for b in %r:
 try:assert psutil.Process(b['pid']).create_time()!=b['birth']
 except psutil.NoSuchProcess:pass
for name,wanted in frozen['files'].items():assert pin(old/name)==wanted,name
for name,wanted in frozen['external'].items():assert pin(Path(name))==wanted,name
for name,wanted in %r.items():assert pin(Path(name))==wanted,name
assert not base.exists();base.mkdir()
for name in frozen['files']:
 if name in ['runtime/supervise.py','prospective-plan.md']:continue
 target=base/name;target.parent.mkdir(parents=True,exist_ok=True);os.link(old/name,target)
print('staged')
'''%(PRIOR_REMOTE,REMOTE,gate['prior_frozen'],receipt['births'],external)
    assert ssh(script).strip()=='staged'
    for path,name in [(folder/'supervise.py','runtime/supervise.py'),(BASE/'prior-gate.json','prior-gate.json'),
                      (PRIOR_BASE/'failure-closed.json','prior-failure-closed.json'),
                      (ROOT/'.agent/m5-audio-amd-two-family-20260920.md','prospective-plan.md')]:
        subprocess.run(['scp','-i',KEY,'-o','BatchMode=yes',str(path),HOST+':'+REMOTE+'/'+name],check=True)
    script='''from pathlib import Path
import hashlib,json
base=Path(%r);old=Path(%r)
def pin(p):
 with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
frozen=json.loads((old/'frozen.json').read_text());frozen['source']=%r
frozen['prior_frozen']=pin(old/'frozen.json');frozen['scope']=['parakeet','pyannote'];frozen['external'].update(%r)
assert pin(base/'runtime/supervise.py')==%r and pin(base/'prior-gate.json')==%r
assert pin(base/'prior-failure-closed.json')==%r
frozen['files']={p.relative_to(base).as_posix():pin(p) for p in sorted(base.rglob('*')) if p.is_file()}
with (base/'frozen.json').open('x') as f:json.dump(frozen,f,indent=2)
print(json.dumps(dict(frozen=pin(base/'frozen.json'),files=len(frozen['files']),external=len(frozen['external']))))
'''%(REMOTE,PRIOR_REMOTE,source,external,pin(folder/'supervise.py'),pin(BASE/'prior-gate.json'),gate['failure_closure'])
    result=json.loads(ssh(script));write(BASE/'freeze-receipt.json',result)
    subprocess.run(['scp','-i',KEY,'-o','BatchMode=yes',HOST+':'+REMOTE+'/frozen.json',str(BASE/'frozen.json')],check=True)
    assert pin(BASE/'frozen.json')==result['frozen']
    launch='''from pathlib import Path
import json,os,subprocess,sys
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
base=Path(%r);assert not (base/'deployment.json').exists() and not (base/'campaign').exists()
frozen=json.loads((base/'frozen.json').read_text())
env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
env.update(PYTHONPATH=os.pathsep.join(frozen['python_paths']),PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1')
env.update({k:'1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS']})
with (base/'supervisor.stdout').open('x') as out,(base/'supervisor.stderr').open('x') as err:
 p=subprocess.Popen(['python3','-B',str(base/'runtime/supervise.py'),str(base)],cwd=base,env=env,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True)
result=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),frozen=%r)
with (base/'deployment.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
'''%(REMOTE,result['frozen'])
    deployment=json.loads(ssh(launch));write(BASE/'deployment.json',deployment);print(json.dumps(deployment))


if __name__=='__main__':main()
