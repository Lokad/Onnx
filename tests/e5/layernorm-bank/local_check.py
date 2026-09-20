"""Exercise actual host loading and complete bank construction, with no kernel calls."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
import numpy as np
from common import BANKS,CORE,pin,read,write
from data import describe
ROOT=Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
import psutil

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--artifact',type=Path,required=True);args=parser.parse_args();base=args.artifact.resolve()
    assert not (base/'local-check.json').exists();origin=ROOT/'artifacts/e5-layernorm-amd-proof-20260920/collected'
    parent=psutil.Process();old=parent.cpu_affinity();all_members={};results={}
    clean={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
    for mode,arguments in [('identity',[]),('inspect',[str(origin),str(base/'payload/banks.json')])]:
        members={};child=None;peak=0
        command=['dotnet',str(base/'payload/bin/LayerNormBank.dll'),mode]+arguments
        try:
            with (base/(mode+'.stdout')).open('x') as stdout,(base/(mode+'.stderr')).open('x') as stderr:
                parent.cpu_affinity([2]);child=subprocess.Popen(command,cwd=ROOT,env=clean,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
                parent.cpu_affinity([0]);birth=psutil.Process(child.pid).create_time();members[str(child.pid)]=birth;start=time.monotonic()
                while child.poll() is None:
                    rss=0
                    try:
                        owner=psutil.Process(child.pid);assert owner.create_time()==birth
                        for process in [owner]+owner.children(recursive=True):
                            try:assert process.cpu_affinity()==[2];members[str(process.pid)]=process.create_time();rss+=process.memory_info().rss
                            except psutil.NoSuchProcess:pass
                    except psutil.NoSuchProcess:pass
                    peak=max(peak,rss);assert time.monotonic()-start<60 and peak<3*1024**3 and psutil.virtual_memory().available>=1024**3
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
        all_members.update(members);value=read(base/(mode+'.stdout'));identity=value if mode=='identity' else value['identity']
        assert identity['mode']==mode and identity['runtime']=='10.0.12' and identity['core_sha256']==CORE
        assert identity['probe_sha256']==pin(base/'payload/bin/LayerNormBank.dll')['sha256'] and identity['settings']=={}
        assert identity['affinity']==4 and identity['vector_width']==8 and not identity['avx512'] and not identity['vector512_hardware']
        if mode=='inspect':assert value['banks']==[describe(origin,d) for d in BANKS]
        results[mode]=dict(command=command,peak_rss=peak,result=value)
    write(base/'local-check.json',dict(passed=True,probe=pin(base/'payload/bin/LayerNormBank.dll'),members=all_members,checks=results,scope='Identity and every mapped input/reference byte; no kernel or model inference'))
    print('Identity and all nine complete bank mappings match independent reconstruction; all births terminal.')

if __name__=='__main__':main()
