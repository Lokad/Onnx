"""Supervise one fixed encoder worker; never replay an existing destination."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,traceback
import numpy as np
from common import ROOT,pin,read,verify
sys.path.append(str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
import psutil

def absent(identity):
    try:return psutil.Process(identity['pid']).create_time()!=identity['birth']
    except psutil.NoSuchProcess:return True

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact',type=Path,required=True);parser.add_argument('--engine',choices=['managed','native'],required=True)
    args=parser.parse_args();base=args.artifact.resolve();manifest=base/'manifest.json';spec=read(manifest);verify(spec)
    assert os.name=='nt' and psutil.virtual_memory().available>=spec['limits']['preflight_available'],'Preflight memory/platform'
    if args.engine=='native':
        previous=read(base/'managed-process/identity.json')
        assert previous['complete'] and previous['code']==0 and absent(previous['supervisor'])
        assert all(absent(dict(pid=int(pid),birth=birth)) for pid,birth in previous['members'].items())
        assert read(base/'managed/result.json')['complete']
    folder=base/(args.engine+'-process');folder.mkdir()
    command=(['dotnet',str(base/'bin/WhisperInputCross.dll'),str(ROOT),str(manifest),str(base/'managed')]
             if args.engine=='managed' else [str(ROOT/'artifacts/asr-labeled-20260919/venv/Scripts/python.exe'),'-X','utf8','-B',
                 str(Path(__file__).with_name('native.py')),'--manifest',str(manifest),'--output',str(base/'native')])
    parent=psutil.Process();old=parent.cpu_affinity();parent.cpu_affinity([0]);child=None
    state=dict(schema=1,engine=args.engine,supervisor=dict(pid=parent.pid,birth=parent.create_time()),started=time.time(),
               manifest_sha256=pin(manifest)['sha256'],complete=False,command=command,limits=spec['limits'],
               preflight_available=psutil.virtual_memory().available,members={},samples=0,peak_rss=0)
    assert state['preflight_available']>=spec['limits']['preflight_available']
    def save():
        temp=folder/'identity.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(folder/'identity.json')
    clean={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
    save()
    try:
        with (folder/'stdout.txt').open('x') as stdout,(folder/'stderr.txt').open('x') as stderr,(folder/'samples.jsonl').open('x') as samples:
            parent.cpu_affinity([2])
            try:child=subprocess.Popen(command,cwd=ROOT,env=clean,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
            finally:parent.cpu_affinity([0])
            birth=psutil.Process(child.pid).create_time();state['child']=dict(pid=child.pid,birth=birth)
            state['members'][str(child.pid)]=birth;start=time.monotonic();save()
            while child.poll() is None:
                members=[]
                try:
                    owner=psutil.Process(child.pid);assert owner.create_time()==birth
                    for process in [owner]+owner.children(recursive=True):
                        try:
                            row=dict(pid=process.pid,birth=process.create_time(),rss=process.memory_info().rss,affinity=process.cpu_affinity())
                            assert row['affinity']==[2]
                            assert str(row['pid']) not in state['members'] or state['members'][str(row['pid'])]==row['birth']
                            state['members'][str(row['pid'])]=row['birth'];members.append(row)
                        except psutil.NoSuchProcess:pass
                except psutil.NoSuchProcess:pass
                sample=dict(seconds=time.monotonic()-start,available=psutil.virtual_memory().available,members=members)
                samples.write(json.dumps(sample)+'\n');samples.flush();state['samples']+=1
                state['peak_rss']=max(state['peak_rss'],sum(r['rss'] for r in members));save()
                assert sample['seconds']<spec['limits']['seconds'] and sample['available']>=spec['limits']['available'] and state['peak_rss']<=spec['limits']['rss'],'Resource guard'
                time.sleep(.5)
            state['code']=child.wait();state['seconds']=time.monotonic()-start
            assert state['code']==0,'Worker execution failure; preserve all evidence'
            assert read(base/args.engine/'result.json')['complete']
            verify(spec);state['complete']=True
    except BaseException:
        state['error']=traceback.format_exc();state['code']=2;raise
    finally:
        for pid,birth in reversed(list(state['members'].items())):
            try:
                process=psutil.Process(int(pid))
                if process.create_time()==birth:process.kill()
            except psutil.NoSuchProcess:pass
        if child is not None:child.wait(timeout=10)
        state['ended']=time.time();state['terminal_members']=all(absent(dict(pid=int(pid),birth=birth)) for pid,birth in state['members'].items())
        save();parent.cpu_affinity(old)
    print(json.dumps(state))

if __name__=='__main__':main()
