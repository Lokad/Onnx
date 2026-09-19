"""Fixed Linux CPU2 prototype campaign, standard-library-only process accounting."""
from pathlib import Path
import hashlib,json,os,signal,subprocess,time,traceback
import campaign_processes as accounting

base=Path(__file__).resolve().parent;os.sched_setaffinity(0,{0});out=base/'result';out.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
    for name,pin in json.loads((base/'bundle.json').read_text())['files'].items():
        p=base/name;assert p.stat().st_size==pin['bytes'] and sha(p)==pin['sha256'],name
def members(group):
    result=[]
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():continue
        try:
            line=(directory/'stat').read_text();fields=line[line.rfind(')')+2:].split()
            if int(fields[2])!=group or fields[0]=='Z':continue
            status=dict(v.split(':',1) for v in (directory/'status').read_text().splitlines() if ':' in v)
            result.append(dict(pid=int(directory.name),start=int(fields[19]),name=line[line.find('(')+1:line.rfind(')')],
                rss=int(status.get('VmRSS','0 kB').split()[0])*1024,affinity=status['Cpus_allowed_list'].strip()))
        except (FileNotFoundError,ProcessLookupError):pass
    return result
def stop(child,birth):
    if child.poll() is None:
        root=next((m for m in members(child.pid) if m['pid']==child.pid),None)
        assert root is not None and root['start']==birth,'Cannot verify owned process before stopping'
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    return child.wait()
identity=dict(supervisor=os.getpid(),supervisor_affinity=sorted(os.sched_getaffinity(0)),runs=[],complete=False)
def save():(out/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
clean={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
code=2
try:
    verify();save();(out/'cpuinfo.txt').write_text(Path('/proc/cpuinfo').read_text());(out/'dotnet-info.txt').write_text(subprocess.check_output(['dotnet','--info'],text=True))
    for index in range(9):
        disassembly=index==8;order=0 if disassembly else index;name='disassembly' if disassembly else f'worker-{order}'
        directory=out/name;directory.mkdir();env=clean.copy();flags={}
        if disassembly:flags.update(DOTNET_JitDisasm='ZeroBlocks.Kernels:Candidate',DOTNET_JitStdOutFile=str(directory/'codegen.txt'));env.update(flags)
        row=dict(index=index,name=name,order=order,disassembly=disassembly,flags=flags,started=time.time(),samples=[],members={},code=None)
        pre=accounting.snapshot();(directory/'pre.json').write_text(json.dumps(pre));(directory/'cpu-pre.txt').write_text(Path('/proc/stat').read_text())
        start=time.monotonic();birth=None
        with (directory/'log.txt').open('x') as log:
            command=['taskset','-c','2','dotnet',str(base/'bin/Probe.dll'),str(directory/'samples.json'),str(order)]
            child=subprocess.Popen(command,cwd=base,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            row.update(pid=child.pid,command=command);identity['runs'].append(row);save()
            try:
                while True:
                    found=next((m for m in members(child.pid) if m['pid']==child.pid),None)
                    if found is not None:birth=found['start']
                    if found is not None and found['name']=='dotnet' and found['affinity']=='2':break
                    assert child.poll() is None and time.monotonic()-start<10,'Worker startup failed';time.sleep(.02)
                row['start_identity']=birth
                while child.poll() is None:
                    group=members(child.pid)
                    for m in group:
                        assert m['affinity']=='2',m;row['members'][str(m['pid'])]=dict(start=m['start'],affinity=m['affinity'],name=m['name'])
                    row['samples'].append(dict(seconds=time.monotonic()-start,members=group));save()
                    assert sum(m['rss'] for m in group)<1024**3 and time.monotonic()-start<180,'Resource guard'
                    time.sleep(.2)
            finally:
                row['code']=stop(child,birth);row.update(seconds=time.monotonic()-start,ended=time.time());save()
        post=accounting.snapshot();(directory/'post.json').write_text(json.dumps(post));(directory/'cpu-post.txt').write_text(Path('/proc/stat').read_text())
        row['accounting']=accounting.foreign_fraction(pre,post,os.getpid());save()
        assert row['code']==0 and not members(child.pid) and not Path('/proc',str(child.pid)).exists(),row
        verify();print(name,'complete',row['seconds'],flush=True)
    identity['complete']=True;save();code=0
except BaseException:
    identity['error']=traceback.format_exc();save();traceback.print_exc()
finally:(base/'complete.txt').write_text(str(code)+'\n')
raise SystemExit(code)
