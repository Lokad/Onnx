"""Fixed twenty-worker AMD managed A/A experiment, using retained process helpers."""
from pathlib import Path
import argparse,importlib.util,json,os,subprocess,time,traceback

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

directory=Path(__file__).resolve().parent
h=module('paired_process_support',directory/'process_support.py' if (directory/'process_support.py').exists() else directory.parents[1]/'parakeet/recording-amd/remote.py')
processes=module('paired_accounting',directory/'campaign_processes.py' if (directory/'campaign_processes.py').exists() else directory.parents[2]/'eng/campaign_processes.py')
h.RSS_LIMIT,h.TIME_LIMIT,h.AVAILABLE_MIN=8*1024**3,600,1024**3

def install(base):
    bundle=h.read(base/'bundle.json');assert not (base/'installed.json').exists()
    allowed=Path('/home/vermorel/Onnx/artifacts/whisper-recording-amd-v2-20260919/bin')
    for name,source in bundle['borrowed'].items():
        source=Path(source);target=h.safe_path(base,name)
        assert not source.is_symlink() and source.resolve().parent==allowed and not target.exists()
        h.verify_file(source,bundle['files'][name]);target.parent.mkdir(parents=True,exist_ok=True);os.link(source,target)
    h.verify(base);h.verify_file(Path(bundle['model']['path']),bundle['model'])
    available=os.statvfs(base);free=available.f_bavail*available.f_frsize
    assert free>=96*1024**2,free
    h.write_new(base/'installed.json',dict(bundle_sha256=h.sha(base/'bundle.json'),available_bytes=free,installed_at=time.time()))
    print('Verified A/A payload, model and at least96MiB result space.',flush=True)

def run(base):
    h.verify(base);bundle=h.read(base/'bundle.json')
    assert h.read(base/'installed.json')['bundle_sha256']==h.sha(base/'bundle.json')
    out=base/'result';out.mkdir();os.sched_setaffinity(0,{0})
    identity=dict(schema=1,supervisor=h.proc(os.getpid()),started=time.time(),complete=False,runs=[],
        bundle_sha256=h.sha(base/'bundle.json'),limits=dict(rss=h.RSS_LIMIT,seconds=h.TIME_LIMIT,available_memory=h.AVAILABLE_MIN))
    def save():
        path=out/'identity.tmp';path.write_text(json.dumps(identity,indent=2)+'\n');path.replace(out/'identity.json')
    code=2
    try:
        save();(out/'cpuinfo.txt').write_text(Path('/proc/cpuinfo').read_text());(out/'dotnet-info.txt').write_text(subprocess.check_output(['dotnet','--info'],text=True))
        clean={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
        for item in bundle['schedule']:
            h.verify(base);name=item['name'];command=['dotnet',str(base/'bin/PairedHost.dll'),bundle['model']['path'],
                str(base/'inputs'/(item['case']+'.json')),str(out/name),str(item['visit']),str(item['case_index']),'timing']
            row=dict(name=name,command=command,started=time.time(),code=None,samples=0,peak_rss=0,members={})
            before=processes.snapshot();h.write_new(out/(name+'-pre.json'),before)
            (out/(name+'-procstat-before.txt')).write_text(Path('/proc/stat').read_text())
            start=time.monotonic()
            with (out/(name+'.stdout')).open('x') as stdout,(out/(name+'.stderr')).open('x') as stderr,(out/(name+'-samples.jsonl')).open('x') as samples:
                os.sched_setaffinity(0,{2})
                try:child=subprocess.Popen(command,cwd=base,env=clean,stdout=stdout,stderr=stderr,start_new_session=True)
                finally:os.sched_setaffinity(0,{0})
                first=h.proc(child.pid);assert first
                row.update(pid=child.pid,start=first['start']);identity['runs'].append(row);save()
                try:
                    while child.poll() is None:
                        sample=dict(seconds=time.monotonic()-start,members=h.members(child.pid),available_memory=h.available())
                        samples.write(json.dumps(sample)+'\n');samples.flush();row['samples']+=1
                        row['peak_rss']=max(row['peak_rss'],sum(m['rss'] for m in sample['members']))
                        for member in sample['members']:row['members'][str(member['pid'])]=member['start']
                        save();h.check_sample(sample,child.pid,first['start']);time.sleep(.5)
                finally:
                    row['code']=h.stop(child,first['start']);row.update(seconds=time.monotonic()-start,ended=time.time());save()
            after=processes.snapshot();h.write_new(out/(name+'-post.json'),after)
            (out/(name+'-procstat-after.txt')).write_text(Path('/proc/stat').read_text())
            row['accounting']=processes.foreign_fraction(before,after,os.getpid());save()
            assert row['code']==0 and row['seconds']<600,name
            print('Complete',name,row['seconds'],row['peak_rss'],flush=True)
        h.verify(base);identity['complete']=True;code=0
    except BaseException:
        identity['error']=traceback.format_exc();traceback.print_exc()
    finally:
        identity['ended']=time.time();save();(base/'complete.txt').write_text(str(code)+'\n')
    return code

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('install','launch','run','collect'));p.add_argument('base',type=Path)
    a=p.parse_args();raise SystemExit(dict(install=install,launch=h.launch,run=run,collect=h.collect)[a.action](a.base.resolve()) or 0)
