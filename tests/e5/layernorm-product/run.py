"""Run eight sequential CPU2 correctness workers, retaining every owned birth and sample."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,time,traceback
import psutil

def pin(path):
    with path.open('rb') as stream:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())

def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--payload',type=Path,required=True)
    parser.add_argument('--assets',type=Path,required=True);parser.add_argument('--label',required=True);args=parser.parse_args()
    base=args.payload.resolve();root=args.assets.resolve();meta=json.loads((base/'frozen.json').read_text())
    assert args.label.replace('-','').isalnum()
    out=base/('result-'+args.label);assert not out.exists();out.mkdir()
    parent=psutil.Process();old_affinity=parent.cpu_affinity();parent.cpu_affinity([0])
    clean={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
    state=dict(schema=1,supervisor=dict(pid=parent.pid,birth=parent.create_time()),started=time.time(),complete=False,
               limits=meta['limits'],frozen=pin(base/'frozen.json'),runs=[])
    save(out/'identity.json',state)
    try:
        for name,wanted in meta['files'].items():assert pin(base/name)==wanted,name
        for name,wanted in meta['assets'].items():assert pin(root/name)==wanted,name
        (out/'dotnet-info.txt').write_text(subprocess.check_output(['dotnet','--info'],cwd=base,text=True),encoding='utf-8')
        if Path('/proc/cpuinfo').exists():(out/'cpuinfo.txt').write_text(Path('/proc/cpuinfo').read_text())
        for job in meta['jobs']:
            if args.label not in job.get('hosts',[args.label]):continue
            folder=out/job['name'];folder.mkdir();kind=job['kind'];child=None
            if kind in ['backend','tensors']:
                dll='Lokad.Onnx.'+('Backend' if kind=='backend' else 'Tensors')+'.Tests.dll'
                command=['dotnet','vstest',str(base/meta['paths'][kind]/dll),'--logger:trx;LogFileName=tests.trx','--ResultsDirectory:'+str(folder)]
            else:
                reference=base/'inputs' if kind=='e5' else root/'artifacts/shared-regression-20260918/reference'
                command=['dotnet',str(base/meta['paths']['replay']/'Replay.dll'),kind,str(root),str(reference),str(folder/'output'),meta['core']['sha256']]
            run=dict(job=job,command=command,started=time.time(),members={},samples=0,peak_rss=0)
            state['runs'].append(run);save(out/'identity.json',state)
            try:
                with (folder/'stdout.txt').open('x') as stdout,(folder/'stderr.txt').open('x') as stderr,(folder/'samples.jsonl').open('x') as samples:
                    parent.cpu_affinity([2])
                    try:
                        settings=clean|{'LOKAD_ONNX_LAYERNORM_WIDE_OUTPUT':str(job['enabled'])}
                        if kind=='code':settings|={'COMPlus_JitDisasm':'*LayerNormFloatInto*','COMPlus_JitStdOutFile':str(folder/'jit.txt')}
                        child=subprocess.Popen(command,cwd=base,env=settings,
                            stdout=stdout,stderr=stderr,**(dict(creationflags=subprocess.CREATE_NO_WINDOW) if os.name=='nt' else dict(start_new_session=True)))
                    finally:parent.cpu_affinity([0])
                    birth=psutil.Process(child.pid).create_time();run['child']=dict(pid=child.pid,birth=birth);run['members'][str(child.pid)]=birth
                    start=time.monotonic();save(out/'identity.json',state)
                    while child.poll() is None:
                        members=[]
                        try:
                            owner=psutil.Process(child.pid);assert owner.create_time()==birth
                            for process in [owner]+owner.children(recursive=True):
                                try:
                                    item=dict(pid=process.pid,birth=process.create_time(),rss=process.memory_info().rss,affinity=process.cpu_affinity())
                                    assert item['affinity']==[2] and item['birth']>=birth
                                    assert str(item['pid']) not in run['members'] or run['members'][str(item['pid'])]==item['birth']
                                    run['members'][str(item['pid'])]=item['birth'];members.append(item)
                                except psutil.NoSuchProcess:pass
                        except psutil.NoSuchProcess:pass
                        sample=dict(seconds=time.monotonic()-start,available=psutil.virtual_memory().available,members=members)
                        samples.write(json.dumps(sample)+'\n');samples.flush();run['samples']+=1
                        run['peak_rss']=max(run['peak_rss'],sum(m['rss'] for m in members));save(out/'identity.json',state)
                        assert sample['seconds']<meta['limits']['seconds'] and sample['available']>=meta['limits']['available'] and run['peak_rss']<meta['limits']['rss'],'Resource guard'
                        time.sleep(.25)
                    run['code']=child.wait();run['seconds']=time.monotonic()-start;assert run['code']==0,job
            finally:
                if child is not None:
                    for pid,birth in reversed(list(run['members'].items())):
                        try:
                            process=psutil.Process(int(pid))
                            if process.create_time()==birth:process.kill()
                        except psutil.NoSuchProcess:pass
                    child.wait(timeout=10)
                run['ended']=time.time();save(out/'identity.json',state)
            print(job['name'],'complete',run['seconds'],flush=True)
        state['complete']=True;state['code']=0
    except BaseException:
        state['error']=traceback.format_exc();state['code']=2;raise
    finally:
        state['ended']=time.time();save(out/'identity.json',state);parent.cpu_affinity(old_affinity)

if __name__=='__main__':main()
