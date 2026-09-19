"""Fixed AMD packed-input pointer proof/comparison; no product or VM checkout changes."""
from pathlib import Path
import argparse,importlib.util,json,os,subprocess,time,traceback,sys

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

directory=Path(__file__).resolve().parent
helper=directory/'process_support.py'
h=module('input_pack_process_support',helper if helper.exists() else directory.parents[1]/'parakeet/recording-amd/remote.py')
account=directory/'campaign_processes.py'
account=account if account.exists() else directory.parents[2]/'eng/campaign_processes.py'
processes=module('input_pack_accounting',account)
h.RSS_LIMIT=2*1024**3
h.TIME_LIMIT=1200

def run(base):
    h.verify(base)
    out=base/'result';out.mkdir()
    os.sched_setaffinity(0,{0})
    identity=dict(schema=1,supervisor=h.proc(os.getpid()),started=time.time(),complete=False,runs=[],
        bundle_sha256=h.sha(base/'bundle.json'),limits=dict(rss=h.RSS_LIMIT,seconds=h.TIME_LIMIT,available_memory=h.AVAILABLE_MIN))
    def save():
        p=out/'identity.tmp';p.write_text(json.dumps(identity,indent=2)+'\n');p.replace(out/'identity.json')
    code=2
    try:
        save()
        (out/'cpuinfo.txt').write_text(Path('/proc/cpuinfo').read_text())
        (out/'dotnet-info.txt').write_text(subprocess.check_output(['dotnet','--info'],text=True))
        clean={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
        phase=h.read(base/'bundle.json')['phase'];assert phase in ('proof','timing')
        for name in (['proof'] if phase=='proof' else ['1','2','3','4']):
            h.verify(base)
            flags={} if name!='proof' else dict(COMPlus_JitDisasm='*PackedTile12* *PackRows12*',COMPlus_JitStdOutFile=str(out/'jit.txt'))
            args=[str(out/'proof.json'),'--proof'] if name=='proof' else [str(out/(name+'.json')),name]
            command=['dotnet',str(base/'bin/Probe.dll')]+args
            row=dict(name=name,command=command,started=time.time(),code=None,samples=0,peak_rss=0,members={},flags=flags)
            before=processes.snapshot();h.write_new(out/(name+'-pre.json'),before)
            start=time.monotonic()
            with (out/(name+'.stdout')).open('x') as stdout,(out/(name+'.stderr')).open('x') as stderr,(out/(name+'-samples.jsonl')).open('x') as samples:
                os.sched_setaffinity(0,{2})
                try:child=subprocess.Popen(command,cwd=base,env=clean|flags,stdout=stdout,stderr=stderr,start_new_session=True)
                finally:os.sched_setaffinity(0,{0})
                first=h.proc(child.pid);assert first
                birth=first['start'];row.update(pid=child.pid,start=birth);identity['runs'].append(row);save()
                try:
                    while child.poll() is None:
                        sample=dict(seconds=time.monotonic()-start,members=h.members(child.pid),available_memory=h.available())
                        samples.write(json.dumps(sample)+'\n');samples.flush()
                        row['samples']+=1;row['peak_rss']=max(row['peak_rss'],sum(m['rss'] for m in sample['members']))
                        for m in sample['members']:row['members'][str(m['pid'])]=m['start']
                        save();h.check_sample(sample,child.pid,birth);time.sleep(.1 if name=='proof' else .25)
                finally:
                    row['code']=h.stop(child,birth);row.update(seconds=time.monotonic()-start,ended=time.time());save()
            after=processes.snapshot();h.write_new(out/(name+'-post.json'),after)
            row['accounting']=processes.foreign_fraction(before,after,os.getpid());save()
            assert row['code']==0 and row['seconds']<h.TIME_LIMIT,name
            print('Complete',name,row['seconds'],row['peak_rss'],flush=True)
        h.verify(base);identity['complete']=True;code=0
    except BaseException:
        identity['error']=traceback.format_exc();traceback.print_exc()
    finally:
        identity['ended']=time.time();save();(base/'complete.txt').write_text(str(code)+'\n')
    return code

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','run','collect']);p.add_argument('base',type=Path)
    a=p.parse_args();raise SystemExit(dict(launch=h.launch,run=run,collect=h.collect)[a.action](a.base.resolve()) or 0)
