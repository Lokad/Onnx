"""Freeze exact old assets/new managed binaries, then launch one bounded campaign."""
import json,subprocess
from common import *

def execute(script):
    compile(script,'checked-whisper-staging','exec');return ssh(script)

def main():
    prepared=read(BASE/'prepared.json');assert prepared['passed']
    for name,wanted in prepared['qualified'].items():assert pin(ROOT/name)==wanted,name
    for name,wanted in prepared['source_bridge'].items():assert pin(ROOT/name)==wanted,name
    for name,wanted in prepared['bin'].items():assert pin(BASE/'bin'/name)==wanted,name
    assert pin(BASE/'manifests/whisper.json')==prepared['manifest']
    assert read(CONTRACTS/'final-verification.json')['passed']
    prior=PRIOR/'collected';original=read(prior/'frozen.json');receipt=read(prior/'collection.json');gate=read(BASE/'prior-native-gate.json')
    old_manifest=read(prior/'manifests/whisper.json');new_manifest=read(BASE/'manifests/whisper.json')
    assert {k:v for k,v in old_manifest.items() if k not in ['core_sha256','data_sha256','product_source']}=={k:v for k,v in new_manifest.items() if k not in ['core_sha256','data_sha256','product_source']}
    native_files={n:w for n,w in receipt['files'].items() if n.startswith(gate['worker']['output']+'/')}
    for name,wanted in native_files.items():assert pin(prior/name)==wanted,name
    # Independently recheck every saved native frontend value before reusing its gate.
    sys.path.insert(0,str(LOCAL_SITE));import numpy as np
    value=read(prior/gate['worker']['output']/'worker/result.json')
    import hashlib
    for case,row in zip(old_manifest['cases'],value['records'],strict=True):
        a=np.load(prior/gate['worker']['output']/'worker'/(case['name']+'.features.npy'),allow_pickle=False)
        b=np.load(prior/'assets'/case['features']['path'],allow_pickle=False)
        assert a.dtype==b.dtype==np.float32 and a.shape==b.shape==(1,128,3000) and np.isfinite(a).all()
        difference=np.abs(a.astype(np.float64)-b.astype(np.float64))
        assert row['frontend']==dict(values=384000,max_abs=float(difference.max()),failed=int((difference>1e-5).sum()),bits_equal=a.tobytes()==b.tobytes(),sha256=hashlib.sha256(a.tobytes()).hexdigest())
        assert row['frontend']['failed']==0
    links={n:w for n,w in original['files'].items() if n.startswith('assets/') or n in ['runtime/native.py','runtime/protocol.py','runtime/whisper_adapter.py','runtime/campaign_processes.py']}
    uploads={}
    for name,wanted in prepared['bin'].items():
        target='bin/'+name
        if original['files'].get(target)==wanted:links[target]=wanted
        else:uploads[target]=BASE/'bin'/name
    uploads.update({'runtime/supervise.py':Path(__file__).parent/'supervise.py','manifests/whisper.json':BASE/'manifests/whisper.json','prior-native-gate.json':BASE/'prior-native-gate.json','prepared.json':BASE/'prepared.json','prospective-plan.md':BASE/'prospective-plan.md'})
    remote_external=dict(original['external'])
    remote_external.update({PRIOR_REMOTE+'/'+n:w for n,w in native_files.items()})
    for n in ['frozen.json','manifests/whisper.json']:remote_external[PRIOR_REMOTE+'/'+n]=pin(prior/n)
    identities=read(CONTRACTS/'closed.json')['births']+receipt['births']
    script=PRELUDE+'''
os.sched_setaffinity(0,{0});terminal(%r)
assert not base.exists()
assert psutil.virtual_memory().available>13*1024**3 and psutil.disk_usage(str(base.parent)).free>64*1024**2
assert pin(old/'frozen.json')==%r
for name,wanted in %r.items():assert pin(old/name)==wanted,name
for name,wanted in %r.items():assert pin(Path(name))==wanted,name
base.mkdir()
for name in %r:
 target=base/name;target.parent.mkdir(parents=True,exist_ok=True);os.link(old/name,target)
for name in %r:(base/name).parent.mkdir(parents=True,exist_ok=True)
print(json.dumps(dict(staged=True,links=%r,available=psutil.virtual_memory().available,disk=psutil.disk_usage(str(base)).free)))
'''%(identities,gate['frozen'],links,remote_external,list(links),list(uploads),len(links))
    write(BASE/'stage.json',json.loads(execute(script)))
    for name,path in uploads.items():subprocess.run(['scp','-i',KEY,'-o','BatchMode=yes',str(path),HOST+':'+REMOTE+'/'+name],check=True)
    upload_pins={n:pin(p) for n,p in uploads.items()}
    frozen={k:v for k,v in original.items() if k not in ['files','external','source','product_source']}
    frozen.update(source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),product_source=prepared['product_source'],scope=['whisper'],external=remote_external,files=links|upload_pins,prior_native=pin(BASE/'prior-native-gate.json'),dotnet='/home/vermorel/.dotnet10.0.8/dotnet')
    script=PRELUDE+'''
os.sched_setaffinity(0,{0});frozen=%r
for name,wanted in frozen['files'].items():assert pin(base/name)==wanted,name
for name,wanted in frozen['external'].items():assert pin(Path(name))==wanted,name
assert str(Path(frozen['dotnet']).resolve()) in frozen['external']
assert {p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file()}==set(frozen['files'])
write(base/'frozen.json',frozen);print(json.dumps(dict(frozen=pin(base/'frozen.json'),files=len(frozen['files']),external=len(frozen['external']))))
'''%frozen
    result=json.loads(execute(script));write(BASE/'freeze-receipt.json',result)
    subprocess.run(['scp','-i',KEY,'-o','BatchMode=yes',HOST+':'+REMOTE+'/frozen.json',str(BASE/'frozen.json')],check=True);assert pin(BASE/'frozen.json')==result['frozen']
    script=PRELUDE+'''
terminal(%r);assert pin(base/'frozen.json')==%r
assert not (base/'deployment.json').exists() and not (base/'campaign').exists()
assert psutil.virtual_memory().available>13*1024**3 and psutil.disk_usage(str(base)).free>64*1024**2
frozen=read(base/'frozen.json')
env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
env.update(PYTHONPATH=os.pathsep.join(frozen['python_paths']),PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1')
env.update({k:'1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS']})
with (base/'supervisor.stdout').open('x') as out,(base/'supervisor.stderr').open('x') as err:
 p=subprocess.Popen(['python3','-B',str(base/'runtime/supervise.py'),str(base)],cwd=base,env=env,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True)
result=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),frozen=pin(base/'frozen.json'));write(base/'deployment.json',result);print(json.dumps(result))
'''%(identities,result['frozen'])
    deployment=json.loads(execute(script));write(BASE/'deployment.json',deployment);print(json.dumps(deployment))

if __name__=='__main__':main()
