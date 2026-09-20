"""Build the consumer once, then freeze committed tools and exact proven kernels."""
import argparse, os, shutil, subprocess, tarfile, time
from common import *

LANE=Path(__file__).resolve().parent
PROVEN=ROOT/'artifacts/e5-reduction-blocks-local-v3-20260920/bin'


def build(base):
    output=base/'build';output.mkdir(parents=True,exist_ok=False);source=output/'source';source.mkdir()
    for name in ['Program.cs','Timing.csproj']:shutil.copyfile(LANE/name,source/name)
    assert pin(PROVEN/'Probe.dll')['sha256']==PROBE and pin(PROVEN/'Lokad.Onnx.dll')['sha256']==CORE
    env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
    commands=[['dotnet','build',str(source/'Timing.csproj'),'-c','Release','--tl:off','--nologo','-v','minimal','--disable-build-servers',
               '-p:FrozenCorePath='+str(PROVEN/'Lokad.Onnx.dll'),'-o',str(output/'bin')],
              ['dotnet',str(output/'bin/Timing.dll'),'--host']]
    jobs=[]
    for i,command in enumerate(commands):
        if i:
            shutil.copyfile(PROVEN/'Probe.dll',output/'bin/Probe.dll')
            assert pin(output/'bin/Lokad.Onnx.dll')['sha256']==CORE
        with (output/f'{i}.stdout').open('x') as stdout,(output/f'{i}.stderr').open('x') as stderr:
            started=time.time();child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
            code=child.wait();jobs.append(dict(pid=child.pid,command=command,started=started,ended=time.time(),code=code))
        assert code==0,(output,i,code)
    host=read(output/'1.stdout');assert not host['supported'] and host['refusals']==3 and host['core']==CORE and host['probe']==PROBE
    write(output/'closed.json',dict(jobs=jobs,host=host,files={p.relative_to(output).as_posix():pin(p) for p in sorted(output.rglob('*')) if p.is_file()}))
    print(json.dumps(dict(host=host,consumer=pin(output/'bin/Timing.dll'),jobs=jobs)))


def freeze(base):
    output=base/'build';closed=read(output/'closed.json')
    for name,wanted in closed['files'].items():assert pin(output/name)==wanted,name
    for name in ['Program.cs','Timing.csproj']:assert pin(LANE/name)==pin(output/'source'/name)
    proof=ROOT/'artifacts/e5-reduction-blocks-proof-20260920';assert pin(proof/'closed.json')['sha256']==PROOF
    receipt=read(proof/'closed.json');assert receipt['passed']
    for name,wanted in receipt['files'].items():assert pin(proof/name)==wanted,name
    for name,wanted in receipt['reports'].items():assert pin(ROOT/name)==wanted,name
    subprocess.run(['git','diff','--exit-code','HEAD','--',str(LANE)],cwd=ROOT,check=True)
    assert not subprocess.check_output(['git','ls-files','--others','--exclude-standard','--',str(LANE)],cwd=ROOT).strip()
    payload=base/'payload';payload.mkdir();write(payload/'schedule.json',schedule());files={}
    for p in LANE.iterdir():
        if p.is_file():files['source/'+p.name]=p
        if p.suffix=='.py':files[p.name]=p
    for p in (output/'bin').iterdir():
        if p.suffix in ['.dll','.json']:files['bin/'+p.name]=p
    files['reference/proof-closed.json']=proof/'closed.json'
    files['reference/build-closed.json']=output/'closed.json'
    files['process_support.py']=ROOT/'tests/parakeet/recording-amd/remote.py'
    files['plan.md']=ROOT/'.agent/m2-reduction-block-timing-20260920.md'
    for name,path in files.items():
        target=payload/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    write(payload/'bundle.json',dict(source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),limits=LIMITS,
          core=CORE,probe=PROBE,proof_receipt=PROOF,files={p.relative_to(payload).as_posix():pin(p) for p in sorted(payload.rglob('*')) if p.is_file()}))
    with tarfile.open(base/'payload.tar.gz','x:gz') as tar:
        for p in sorted(payload.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(payload).as_posix(),recursive=False)
    result=dict(bundle=pin(payload/'bundle.json'),archive=pin(base/'payload.tar.gz'));write(base/'prepared.json',result);print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','freeze']);p.add_argument('--artifact',type=Path,required=True);a=p.parse_args();globals()[a.action](a.artifact.resolve())
