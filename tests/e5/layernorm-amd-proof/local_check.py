"""Exercise only new host identity/loading, without repeating arithmetic inference."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
import numpy as np
from common import CORE,pin,read,write
ROOT=Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
import psutil

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--artifact',type=Path,required=True);args=parser.parse_args();base=args.artifact.resolve()
    assert not (base/'local-identity.json').exists();parent=psutil.Process();old=parent.cpu_affinity();members={}
    clean={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
    command=['dotnet',str(base/'payload/bin/LayerNormAmdProof.dll'),'identity'];child=None
    try:
        with (base/'identity.stdout').open('x') as stdout,(base/'identity.stderr').open('x') as stderr:
            parent.cpu_affinity([2]);child=subprocess.Popen(command,cwd=ROOT,env=clean,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
            parent.cpu_affinity([0]);birth=psutil.Process(child.pid).create_time();members[str(child.pid)]=birth;started=time.monotonic()
            while child.poll() is None:
                assert time.monotonic()-started<30
                try:
                    owner=psutil.Process(child.pid);assert owner.create_time()==birth
                    for process in [owner]+owner.children(recursive=True):
                        try:assert process.cpu_affinity()==[2];members[str(process.pid)]=process.create_time()
                        except psutil.NoSuchProcess:pass
                except psutil.NoSuchProcess:pass
                time.sleep(.05)
            assert child.wait()==0
    finally:
        for pid,birth in reversed(list(members.items())):
            try:
                process=psutil.Process(int(pid))
                if process.create_time()==birth:process.kill()
            except psutil.NoSuchProcess:pass
        if child is not None:child.wait(timeout=10)
        parent.cpu_affinity(old)
    for pid,birth in members.items():
        try:assert psutil.Process(int(pid)).create_time()!=birth
        except psutil.NoSuchProcess:pass
    value=json.loads((base/'identity.stdout').read_text());assert value['mode']=='identity' and value['runtime']=='10.0.12'
    assert value['core_sha256']==CORE and value['probe_sha256']==pin(base/'payload/bin/LayerNormAmdProof.dll')['sha256']
    assert value['affinity']==4 and value['vector_width']==8 and value['settings']=={} and value['avx512'] is False and value['vector512_hardware'] is False
    write(base/'local-identity.json',dict(passed=True,identity=value,members=members,command=command,scope='Host loading/settings only; no repeated kernel/model proof'))
    print('Local identity path passes; every observed child birth is terminal.')

if __name__=='__main__':main()
