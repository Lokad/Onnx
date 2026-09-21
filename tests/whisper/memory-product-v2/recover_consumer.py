"""Correct only the synthetic consumer metadata; reuse successful builds and package."""
import os,shutil,subprocess,time,traceback,xml.etree.ElementTree as ET
from common import *

BASE=ROOT/'artifacts/whisper-memory-product-v2-20260921'


def main():
    old=read(BASE/'run.json');assert old['complete'] and old['code']==1
    assert [r['name'] for r in old['runs']]==['solution-build','tensor-tests','backend-tests','pack','restore','build','default']
    assert all(r['code']==0 for r in old['runs'][:-1]) and old['runs'][-1]['code']!=0
    births=[old['supervisor']]+[dict(pid=int(p),birth=b) for r in old['runs'] for p,b in r['members'].items()]
    assert all(absent(b) for b in births)
    error=(BASE/'default.stderr').read_text();assert "The given key 'Name' was not present" in error
    folder=Path(__file__).parent;app=BASE/'consumer-corrected';app.mkdir();tool=BASE/'il-check';tool.mkdir()
    source=(BASE/'consumer/Program.cs').read_text(encoding='utf-8');marker='    var chain = new ComputationalGraph();';assert source.count(marker)==1
    source=source.replace(marker,marker+'\n    chain.Metadata["Name"] = "private-package-released-budget";')
    (app/'Program.cs').write_text(source,encoding='utf-8')
    for name in ['Consumer.csproj','nuget.config']:shutil.copyfile(BASE/'consumer'/name,app/name)
    for name in ['CompareIl.cs','CompareIl.csproj']:shutil.copyfile(folder/name,tool/name)
    files=[app/'Program.cs',app/'Consumer.csproj',app/'nuget.config',tool/'CompareIl.cs',tool/'CompareIl.csproj',Path(__file__),BASE/'run.json',BASE/'default.stderr',BASE/'source/artifacts/nuget/Lokad.Onnx.0.2.0.nupkg']
    frozen=dict(prior_failure=True,births=births,files={str(p.relative_to(ROOT)):pin(p) for p in files})
    write(BASE/'consumer-corrected-frozen.json',frozen)
    ps=psutil_module();parent=ps.Process();affinity=parent.cpu_affinity();parent.cpu_affinity([0])
    state=dict(complete=False,code=None,started=time.time(),frozen=pin(BASE/'consumer-corrected-frozen.json'),supervisor=dict(pid=parent.pid,birth=parent.create_time()),runs=[])
    def save():
        path=BASE/'consumer-corrected-run.tmp';path.write_text(json.dumps(state,indent=2));path.replace(BASE/'consumer-corrected-run.json')
    save()
    clean={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
    clean.update(DOTNET_PROCESSOR_COUNT='1',UseSharedCompilation='false',MSBUILDDISABLENODEREUSE='1',DOTNET_CLI_DO_NOT_USE_MSBUILD_SERVER='1',NUGET_PACKAGES=str(BASE/'packages-corrected'))
    def launch(name,command,cwd,env):
        record=dict(name=name,command=command,started=time.time(),members={},samples=0,complete=False,code=None);state['runs'].append(record);save();child=None;start=time.monotonic()
        try:
            prefix=BASE/('consumer-corrected-'+name)
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
        launch('restore',['dotnet','restore','Consumer.csproj','--configfile','nuget.config',*flags],app,clean)
        launch('build',['dotnet','build','Consumer.csproj','-c','Release','--no-restore',*flags],app,clean)
        dll=app/'bin/Release/net10.0/LayerNormPackageConsumer.dll';fixture=BASE/'source/tests/Lokad.Onnx.Backend.Tests/models/mnist-8.onnx'
        for setting in SETTINGS:
            options={}
            if setting['fingerprint']:options['LOKAD_ONNX_FINGERPRINT_STRINGS']='1'
            if setting['wide']:options['LOKAD_ONNX_LAYERNORM_WIDE_OUTPUT']='1'
            launch(setting['name'],['dotnet',str(dll),str(fixture),str(BASE/('consumer-corrected-'+setting['name']+'.json')),str(int(setting['fingerprint'])),str(int(setting['wide']))],app,clean|options)
        launch('il-build',['dotnet','build','CompareIl.csproj','-c','Release',*flags],tool,clean)
        launch('il-check',['dotnet',str(tool/'bin/Release/net10.0/CompareIl.dll'),str(ROOT/'artifacts/whisper-weight-sharing-20260920/product-bin'),str(BASE/'source/src/Lokad.Onnx.CLI/bin/Release/net10.0'),str(BASE/'method-equivalence.json')],tool,clean)
        for name,wanted in frozen['files'].items():assert pin(ROOT/name)==wanted,name
        state['code']=0
    except BaseException:state['code']=1;state['error']=traceback.format_exc();raise
    finally:state.update(complete=True,ended=time.time());parent.cpu_affinity(affinity);save()


if __name__=='__main__':main()
