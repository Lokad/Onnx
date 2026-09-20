"""Run the full corpus with one finite encoder process per engine and recording."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,traceback
import numpy as np
from protocol import schedule,coverage
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'input-cross'))
from common import ROOT,pin,read,verify
from audit import absent
import psutil

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--artifact',type=Path,required=True);args=parser.parse_args();base=args.artifact.resolve()
    manifest=base/'manifest.json';spec=read(manifest);verify(spec);jobs=schedule(spec);coverage(jobs,spec)
    assert os.name=='nt' and not (base/'campaign.json').exists()
    (base/'process').mkdir();(base/'outputs').mkdir();parent=psutil.Process();old=parent.cpu_affinity();parent.cpu_affinity([0])
    state=dict(schema=1,supervisor=dict(pid=parent.pid,birth=parent.create_time()),started=time.time(),complete=False,runs=[],manifest_sha256=pin(manifest)['sha256'])
    def save():
        temp=base/'campaign.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(base/'campaign.json')
    clean={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))};save()
    try:
        with (base/'preflight.jsonl').open('x') as preflight:
            for job in jobs:
                state['current']=job;state['waiting_for_memory']=True;save();wait_start=time.monotonic()
                while True:
                    available=psutil.virtual_memory().available
                    preflight.write(json.dumps(dict(job=job['id'],time=time.time(),available=available))+'\n');preflight.flush()
                    if available>=spec['limits']['preflight_available']:break
                    assert time.monotonic()-wait_start<spec['preflight_wait_seconds'],'Preflight headroom wait exceeded; no worker replay'
                    time.sleep(5)
                state['waiting_for_memory']=False
                folder=base/'process'/job['id'];folder.mkdir();output=base/'outputs'/job['id']
                command=(['dotnet',str(base/'bin/WhisperInputCross.dll'),str(ROOT),str(manifest),str(output),str(job['request'])]
                    if job['engine']=='managed' else [str(ROOT/'artifacts/asr-labeled-20260919/venv/Scripts/python.exe'),'-X','utf8','-B',
                        str(Path(__file__).with_name('native.py')),'--manifest',str(manifest),'--output',str(output),'--request-index',str(job['request'])])
                run=dict(job=job,supervisor=state['supervisor'],started=time.time(),complete=False,command=command,limits=spec['limits'],
                         preflight_available=available,members={},samples=0,peak_rss=0);state['runs'].append(run);save();child=None
                try:
                    with (folder/'stdout.txt').open('x') as stdout,(folder/'stderr.txt').open('x') as stderr,(folder/'samples.jsonl').open('x') as samples:
                        parent.cpu_affinity([2])
                        try:child=subprocess.Popen(command,cwd=ROOT,env=clean,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
                        finally:parent.cpu_affinity([0])
                        birth=psutil.Process(child.pid).create_time();run['child']=dict(pid=child.pid,birth=birth);run['members'][str(child.pid)]=birth
                        start=time.monotonic();save()
                        while child.poll() is None:
                            members=[]
                            try:
                                owner=psutil.Process(child.pid);assert owner.create_time()==birth
                                for process in [owner]+owner.children(recursive=True):
                                    try:
                                        row=dict(pid=process.pid,birth=process.create_time(),rss=process.memory_info().rss,affinity=process.cpu_affinity())
                                        assert row['affinity']==[2] and (str(row['pid']) not in run['members'] or run['members'][str(row['pid'])]==row['birth'])
                                        run['members'][str(row['pid'])]=row['birth'];members.append(row)
                                    except psutil.NoSuchProcess:pass
                            except psutil.NoSuchProcess:pass
                            sample=dict(seconds=time.monotonic()-start,available=psutil.virtual_memory().available,members=members)
                            samples.write(json.dumps(sample)+'\n');samples.flush();run['samples']+=1;run['peak_rss']=max(run['peak_rss'],sum(r['rss'] for r in members));save()
                            assert sample['seconds']<spec['limits']['seconds'] and sample['available']>=spec['limits']['available'] and run['peak_rss']<=spec['limits']['rss'],'Resource guard'
                            time.sleep(.5)
                        run['code']=child.wait();run['seconds']=time.monotonic()-start
                        assert run['code']==0,'Worker failed; preserve prefix'
                        result=read(output/'result.json')
                        assert result['complete'] is True and result['engine']==job['engine'] and result['request_index']==job['request']
                        assert result['manifest_sha256']==state['manifest_sha256'] and len(result['records'])==2
                        assert result['records'][0]['baseline_matches'] is True,'Baseline failed; do not advance'
                        run['complete']=True
                except BaseException:run['error']=traceback.format_exc();run['code']=2;raise
                finally:
                    for pid,birth in reversed(list(run['members'].items())):
                        try:
                            process=psutil.Process(int(pid))
                            if process.create_time()==birth:process.kill()
                        except psutil.NoSuchProcess:pass
                    if child is not None:child.wait(timeout=10)
                    run['ended']=time.time();run['terminal_members']=all(absent(dict(pid=int(pid),birth=birth)) for pid,birth in run['members'].items());save()
                assert run['terminal_members'],'Original worker birth still live'
                print('Completed',job['id'],flush=True)
        verify(spec);state['complete']=True;state['code']=0
    except BaseException:state['error']=traceback.format_exc();state['code']=2;raise
    finally:state['ended']=time.time();save();parent.cpu_affinity(old)
    print('Completed all 42 finite workers; numerical acceptance requires full independent audit.')

if __name__=='__main__':main()
