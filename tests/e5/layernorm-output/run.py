"""Bound local capture/proof processes without altering the running AMD experiment."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys,time,traceback
import numpy
ROOT=Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
import psutil

def pin(p):
    with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--mode',choices=['capture','proof'],required=True);a=p.parse_args()
    base=a.artifact.resolve();binary=base/'bin-final/LayerNormOutput.dll';model=ROOT/'models/multilingual-e5-small/model.onnx'
    inputs=ROOT/'artifacts/e5-fingerprint-product-v2-20260920/payload/inputs'
    folder=base/(a.mode+'-process');assert not folder.exists();folder.mkdir()
    assert not (base/a.mode).exists()
    command=['dotnet',str(binary),a.mode]+([str(model),str(inputs),str(base/'capture')] if a.mode=='capture' else [str(base/'capture'),str(base/'proof')])
    parent=psutil.Process();old=parent.cpu_affinity();child=None
    state=dict(mode=a.mode,command=command,started=time.time(),complete=False,supervisor=dict(pid=parent.pid,birth=parent.create_time()),members={},samples=0,peak_rss=0,
        limits=dict(seconds=180,rss=6*1024**3,available=2*1024**3),binaries={p.name:pin(p) for p in (base/'bin-final').iterdir() if p.is_file()},
        source={p.name:pin(p) for p in Path(__file__).parent.iterdir() if p.is_file()},generated={p.name:pin(p) for p in (base/'generated').iterdir() if p.is_file()})
    def save():
        temp=folder/'identity.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(folder/'identity.json')
    clean={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
    save()
    try:
        with (folder/'stdout.txt').open('x') as out,(folder/'stderr.txt').open('x') as err,(folder/'samples.jsonl').open('x') as samples:
            parent.cpu_affinity([2])
            try:child=subprocess.Popen(command,cwd=ROOT,env=clean,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            finally:parent.cpu_affinity([0])
            birth=psutil.Process(child.pid).create_time();state['child']=dict(pid=child.pid,birth=birth);state['members'][str(child.pid)]=birth;save();start=time.monotonic()
            while child.poll() is None:
                members=[]
                try:
                    owner=psutil.Process(child.pid);assert owner.create_time()==birth
                    for process in [owner]+owner.children(recursive=True):
                        try:
                            item=dict(pid=process.pid,birth=process.create_time(),rss=process.memory_info().rss,affinity=process.cpu_affinity());assert item['affinity']==[2]
                            assert str(item['pid']) not in state['members'] or state['members'][str(item['pid'])]==item['birth']
                            state['members'][str(item['pid'])]=item['birth'];members.append(item)
                        except psutil.NoSuchProcess:pass
                except psutil.NoSuchProcess:pass
                row=dict(seconds=time.monotonic()-start,available=psutil.virtual_memory().available,members=members)
                samples.write(json.dumps(row)+'\n');samples.flush();state['samples']+=1;state['peak_rss']=max(state['peak_rss'],sum(i['rss'] for i in members));save()
                assert row['seconds']<180 and row['available']>=2*1024**3 and state['peak_rss']<6*1024**3,'Resource guard'
                time.sleep(.25)
            state['code']=child.wait();assert state['code']==0,a.mode
            state['complete']=True
    except BaseException:state['error']=traceback.format_exc();raise
    finally:
        if child is not None:
            for pid,birth in reversed(list(state['members'].items())):
                try:
                    process=psutil.Process(int(pid))
                    if process.create_time()==birth:process.kill()
                except psutil.NoSuchProcess:pass
            child.wait(timeout=10)
        state['ended']=time.time();save();parent.cpu_affinity(old)
    print((folder/'stdout.txt').read_text());print(a.mode,'terminal; peak RSS',state['peak_rss'])

if __name__=='__main__':main()
