"""Fixed AMD code proof and four-worker complete GELU bank experiment."""
from pathlib import Path
import argparse,importlib.util,json,os,subprocess,sys,tarfile,time,traceback

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
directory=Path(__file__).resolve().parent
h=module('gelu_process_support',directory/'process_support.py')
account=module('gelu_accounting',directory/'campaign_processes.py')
h.RSS_LIMIT=3*1024**3;h.TIME_LIMIT=600;h.AVAILABLE_MIN=1024**3

def run(base,code=False):
    h.verify(base);out=base/('code' if code else 'result');out.mkdir();os.sched_setaffinity(0,{0})
    if not code:
        gate=h.read(base/'code-gate.json');assert gate['passed']
        assert gate['jit_sha256']==h.sha(base/'code/jit.txt') and gate['proof_sha256']==h.sha(base/'code/proof.json')
        assert gate['probe_sha256']==h.sha(base/'bin/Probe.dll')
    identity=dict(schema=1,supervisor=h.proc(os.getpid()),started=time.time(),complete=False,runs=[],bundle_sha256=h.sha(base/'bundle.json'),limits=dict(rss=h.RSS_LIMIT,seconds=h.TIME_LIMIT,available_memory=h.AVAILABLE_MIN))
    def save():
        p=out/'identity.tmp';p.write_text(json.dumps(identity,indent=2)+'\n');p.replace(out/'identity.json')
    result=2
    try:
        save();(out/'cpuinfo.txt').write_text(Path('/proc/cpuinfo').read_text());(out/'dotnet-info.txt').write_text(subprocess.check_output(['dotnet','--info'],text=True))
        clean={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
        for name in (['proof'] if code else ['0','1','2','3']):
            h.verify(base);flags=dict(COMPlus_JitDisasm='Kernels:Conditional Kernels:CopyA',COMPlus_JitStdOutFile=str(out/'jit.txt')) if code else {}
            command=['dotnet',str(base/'bin/Probe.dll'),str(base/'data'),str(out/(name+'.json'))]+([] if code else ['--timing',name])
            row=dict(name=name,command=command,started=time.time(),code=None,samples=0,peak_rss=0,members={},flags=flags)
            before=account.snapshot();h.write_new(out/(name+'-pre.json'),before);start=time.monotonic()
            with (out/(name+'.stdout')).open('x') as stdout,(out/(name+'.stderr')).open('x') as stderr,(out/(name+'-samples.jsonl')).open('x') as samples:
                os.sched_setaffinity(0,{2})
                try:child=subprocess.Popen(command,cwd=base,env=clean|flags,stdout=stdout,stderr=stderr,start_new_session=True)
                finally:os.sched_setaffinity(0,{0})
                first=h.proc(child.pid);assert first;birth=first['start'];row.update(pid=child.pid,start=birth);identity['runs'].append(row);save()
                try:
                    while child.poll() is None:
                        sample=dict(seconds=time.monotonic()-start,members=h.members(child.pid),available_memory=h.available())
                        samples.write(json.dumps(sample)+'\n');samples.flush();row['samples']+=1
                        row['peak_rss']=max(row['peak_rss'],sum(m['rss'] for m in sample['members']))
                        for m in sample['members']:row['members'][str(m['pid'])]=m['start']
                        save();h.check_sample(sample,child.pid,birth);time.sleep(.1 if code else .25)
                finally:row['code']=h.stop(child,birth);row.update(seconds=time.monotonic()-start,ended=time.time());save()
            after=account.snapshot();h.write_new(out/(name+'-post.json'),after);row['accounting']=account.foreign_fraction(before,after,os.getpid());save()
            assert row['code']==0 and row['seconds']<600,name
            proof=h.read(out/(name+'.json'));assert proof['passed'] and proof['cases']==1575 and proof['compared']==102364884 and proof['avx512'] and proof['runtime']=='.NET 10.0.8'
            print('Complete',name,row['seconds'],row['peak_rss'],flush=True)
        h.verify(base);identity['complete']=True;result=0
    except BaseException:identity['error']=traceback.format_exc();traceback.print_exc()
    finally:identity['ended']=time.time();save();(out/'complete.txt').write_text(str(result)+'\n')
    return result

def collect(base):
    h.verify(base);roots=[]
    for phase in ['code','result']:
        value=h.read(base/phase/'identity.json');assert value['complete'] and not value.get('error')
        roots.append(value['supervisor'])
        for row in value['runs']:
            roots.append(dict(pid=row['pid'],start=row['start']))
            roots.extend(dict(pid=int(pid),start=birth) for pid,birth in row['members'].items())
    for item in roots:
        live=h.proc(item['pid']);assert live is None or live['start']!=item['start'],'Observed process remains'
    assert not (base/'collection.json').exists()
    files=[p for p in sorted(base.rglob('*')) if p.is_file() and p.relative_to(base).parts[0] not in ('bin','data')]
    inventory={p.relative_to(base).as_posix():dict(bytes=p.stat().st_size,sha256=h.sha(p)) for p in files}
    retained={n:v for n,v in h.read(base/'bundle.json')['files'].items() if n.split('/')[0] in ('bin','data')}
    value=dict(schema=1,passed=True,terminal_processes=roots,files=inventory,verified_reusable_files=retained,scope='All new results and source; bin/data reverified remotely and retained locally')
    h.write_new(base/'collection.json',value);archive=base.with_name(base.name+'-results.tar.gz')
    with tarfile.open(archive,'x:gz') as tar:
        for p in files+[base/'collection.json']:tar.add(p,arcname=p.relative_to(base).as_posix(),recursive=False)
    print(json.dumps(dict(bytes=archive.stat().st_size,sha256=h.sha(archive),collection_sha256=h.sha(base/'collection.json'))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['code','launch','run','collect']);p.add_argument('base',type=Path);a=p.parse_args();base=a.base.resolve()
    result=run(base,True) if a.action=='code' else run(base) if a.action=='run' else h.launch(base) if a.action=='launch' else collect(base)
    raise SystemExit(result or 0)
