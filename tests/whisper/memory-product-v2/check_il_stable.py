"""Correct only the synthetic consumer metadata; reuse successful builds and package."""
import os,shutil,subprocess,time,traceback,xml.etree.ElementTree as ET
from common import *

BASE=ROOT/'artifacts/whisper-memory-product-v2-20260921'


def main():
    old=read(BASE/'consumer-corrected-run.json');assert old['complete'] and old['code']==1
    assert old['runs'][-1]['name']=='il-check' and all(r['code']==0 for r in old['runs'][:-1])
    births=[old['supervisor']]+[dict(pid=int(p),birth=b) for r in old['runs'] for p,b in r['members'].items()]
    assert all(absent(b) for b in births)
    assert 'AssemblyLoadContext is unloading or was already unloaded' in (BASE/'consumer-corrected-il-check.stderr').read_text()
    folder=Path(__file__).parent;tool=BASE/'il-check-stable';tool.mkdir()
    for name in ['CompareIlStable.cs','CompareIlStable.csproj']:shutil.copyfile(folder/name,tool/name)
    frozen=dict(prior_failure=True,files={str(p.relative_to(ROOT)):pin(p) for p in [tool/'CompareIlStable.cs',tool/'CompareIlStable.csproj',Path(__file__),BASE/'consumer-corrected-run.json',BASE/'consumer-corrected-il-check.stderr']})
    write(BASE/'il-stable-frozen.json',frozen)
    ps=psutil_module();parent=ps.Process();affinity=parent.cpu_affinity();parent.cpu_affinity([0])
    state=dict(complete=False,code=None,started=time.time(),frozen=pin(BASE/'il-stable-frozen.json'),supervisor=dict(pid=parent.pid,birth=parent.create_time()),runs=[])
    def save():
        path=BASE/'il-stable-run.tmp';path.write_text(json.dumps(state,indent=2));path.replace(BASE/'il-stable-run.json')
    save()
    clean={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
    clean.update(DOTNET_PROCESSOR_COUNT='1',UseSharedCompilation='false',MSBUILDDISABLENODEREUSE='1',DOTNET_CLI_DO_NOT_USE_MSBUILD_SERVER='1',NUGET_PACKAGES=str(BASE/'packages-corrected'))
    def launch(name,command,cwd,env):
        record=dict(name=name,command=command,started=time.time(),members={},samples=0,complete=False,code=None);state['runs'].append(record);save();child=None;start=time.monotonic()
        try:
            prefix=BASE/('il-stable-'+name)
            with Path(str(prefix)+'.stdout').open('x') as out,Path(str(prefix)+'.stderr').open('x') as err,Path(str(prefix)+'.samples.jsonl').open('x') as samples:
                child=subprocess.Popen(command,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
                process=ps.Process(child.pid);record['child']=dict(pid=child.pid,birth=process.create_time());record['members'][str(child.pid)]=process.create_time();save()
                while child.poll() is None:
                    members=[]
                    try:
                        assert process.create_time()==record['child']['birth']
                        for member in [process]+process.children(recursive=True):
                            try:
                                birth=member.create_time();assert record['members'].get(str(member.pid),birth)==birth;record['members'][str(member.pid)]=birth
                                members.append(dict(pid=member.pid,birth=birth,rss=member.memory_info().rss,affinity=member.cpu_affinity()))
                            except ps.NoSuchProcess:pass
                    except ps.NoSuchProcess:pass
                    row=dict(seconds=time.monotonic()-start,available=ps.virtual_memory().available,members=members)
                    samples.write(json.dumps(row)+'\n');samples.flush();record['samples']+=1;save()
                    assert row['seconds']<300 and row['available']>=1024**3 and sum(m['rss'] for m in members)<4*1024**3 and all(m['affinity']==[0] for m in members)
                    time.sleep(.25)
                record['code']=child.wait();assert record['code']==0,(name,record['code'])
                until=time.monotonic()+10
                while not all(absent(dict(pid=int(p),birth=b)) for p,b in record['members'].items()):
                    assert time.monotonic()<until;time.sleep(.1)
        except BaseException:
            record['error']=traceback.format_exc()
            for pid,birth in reversed(list(record['members'].items())):
                try:
                    p=ps.Process(int(pid))
                    if p.create_time()==birth:p.kill()
                except ps.NoSuchProcess:pass
            if child is not None:child.wait(timeout=10)
            raise
        finally:record.update(complete=True,ended=time.time(),seconds=time.monotonic()-start);save()
        print(json.dumps(dict(stage=name,code=0,seconds=record['seconds'])),flush=True)
    try:
        flags=['--tl:off','--nologo','-v','minimal']
        launch('build',['dotnet','build','CompareIlStable.csproj','-c','Release',*flags],tool,clean)
        launch('check',['dotnet',str(tool/'bin/Release/net10.0/CompareIlStable.dll'),str(ROOT/'artifacts/whisper-weight-sharing-20260920/product-bin'),str(BASE/'source/src/Lokad.Onnx.CLI/bin/Release/net10.0'),str(BASE/'method-equivalence-stable.json')],tool,clean)
        for name,wanted in frozen['files'].items():assert pin(ROOT/name)==wanted,name
        state['code']=0
    except BaseException:state['code']=1;state['error']=traceback.format_exc();raise
    finally:state.update(complete=True,ended=time.time());parent.cpu_affinity(affinity);save()


if __name__=='__main__':main()
