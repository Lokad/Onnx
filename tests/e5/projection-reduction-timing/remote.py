"""Four finite normal-runtime workers, with immutable inputs and Linux start ticks."""
from pathlib import Path
import argparse, importlib.util, json, os, subprocess, time, traceback

spec=importlib.util.spec_from_file_location('process_support',Path(__file__).with_name('process_support.py'))
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
h.TIME_LIMIT=600;h.RSS_LIMIT=2*1024**3;h.AVAILABLE_MIN=1024**3


def cpu():
    return {parts[0]:list(map(int,parts[1:])) for line in Path('/proc/stat').read_text().splitlines() if (parts:=line.split())[0] in ['cpu','cpu2']}


def processes():
    return [item for p in Path('/proc').iterdir() if p.name.isdigit() and (item:=h.proc(p.name))]


def run(base):
    h.verify(base);out=base/'result';out.mkdir();os.sched_setaffinity(0,{0})
    state=dict(supervisor=h.proc(os.getpid()),started=time.time(),complete=False,code=None,runs=[],
        bundle_sha256=h.sha(base/'bundle.json'),limits=dict(seconds=h.TIME_LIMIT,rss=h.RSS_LIMIT,available_memory=h.AVAILABLE_MIN))
    def save():
        p=out/'identity.tmp';p.write_text(json.dumps(state,indent=2));p.replace(out/'identity.json')
    try:
        save();(out/'cpuinfo.txt').write_text(Path('/proc/cpuinfo').read_text())
        (out/'dotnet-info.txt').write_text(subprocess.check_output(['dotnet','--info'],text=True))
        clean={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
        for worker in range(4):
            name=f'worker{worker}';h.verify(base);available=h.available();disk=os.statvfs(base).f_bavail*os.statvfs(base).f_frsize
            assert available>=4*1024**3 and disk>=128*1024**2
            command=['dotnet',str(base/'bin/Timing.dll'),str(base/'schedule.json'),str(worker),str(out/name)]
            row=dict(name=name,command=command,flags={},started=time.time(),code=None,samples=0,peak_rss=0,members={},preflight_available=available,preflight_disk=disk)
            before=processes();row['processes_before']=before;state['runs'].append(row);save();child=None;birth=None;start=time.monotonic()
            try:
                with (out/(name+'.stdout')).open('x') as stdout,(out/(name+'.stderr')).open('x') as stderr,(out/(name+'-samples.jsonl')).open('x') as samples:
                    os.sched_setaffinity(0,{2})
                    try:child=subprocess.Popen(command,cwd=base,env=clean,stdout=stdout,stderr=stderr,start_new_session=True)
                    finally:os.sched_setaffinity(0,{0})
                    first=h.proc(child.pid);assert first;birth=first['start'];row.update(pid=child.pid,start=birth);row['members'][str(child.pid)]=birth;save()
                    while child.poll() is None:
                        sample=dict(seconds=time.monotonic()-start,members=h.members(child.pid),available_memory=h.available(),cpu=cpu())
                        samples.write(json.dumps(sample)+'\n');samples.flush();row['samples']+=1
                        row['peak_rss']=max(row['peak_rss'],sum(m['rss'] for m in sample['members']))
                        for member in sample['members']:
                            assert row['members'].get(str(member['pid']),member['start'])==member['start'];row['members'][str(member['pid'])]=member['start']
                        save();h.check_sample(sample,child.pid,birth);time.sleep(.1)
            finally:
                if child is not None:row['code']=h.stop(child,birth)
                row.update(ended=time.time(),seconds=time.monotonic()-start,processes_after=processes())
                prior={(p['pid'],p['start']):p for p in before};foreign=[]
                for p in row['processes_after']:
                    old=prior.get((p['pid'],p['start']))
                    if old and p['pid']!=state['supervisor']['pid'] and str(p['pid']) not in row['members']:
                        delta=p['cpu_seconds']-old['cpu_seconds']
                        if delta>0:foreign.append(dict(pid=p['pid'],start=p['start'],cpu_seconds=delta,affinity=p['affinity']))
                row['foreign_activity']=foreign;save()
            assert row['code']==0 and row['seconds']<600
            print(json.dumps(dict(name=name,seconds=row['seconds'],samples=row['samples'],peak=row['peak_rss'])),flush=True)
        h.verify(base);state['code']=0
    except BaseException:
        state['error']=traceback.format_exc();state['code']=1;traceback.print_exc()
    finally:
        state.update(complete=True,ended=time.time());save();(base/'complete.txt').write_text(str(state['code'])+'\n')
    return state['code']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','run','collect']);p.add_argument('base',type=Path);a=p.parse_args()
    raise SystemExit(dict(launch=h.launch,run=run,collect=h.collect)[a.action](a.base.resolve()) or 0)
