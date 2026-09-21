"""Small package/build/consumer stages on CPU0 while existing diagnostics run."""
import argparse,os,shutil,subprocess,time,traceback
from common import *

def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');tmp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',required=True);a=p.parse_args();base=Path(a.artifact).resolve();prepared=read(base/'prepared.json')
    assert prepared['settings']==SETTINGS
    for name,wanted in prepared['tools'].items():assert pin(ROOT/name)==wanted,name
    ps=psutil_module();parent=ps.Process();old_affinity=parent.cpu_affinity();parent.cpu_affinity([0]);state_path=base/'run.json';assert not state_path.exists()
    state=dict(supervisor=dict(pid=parent.pid,birth=parent.create_time()),prepared=pin(base/'prepared.json'),started=time.time(),complete=False,code=None,runs=[]);save(state_path,state)
    clean={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
    clean|=dict(DOTNET_PROCESSOR_COUNT='1',UseSharedCompilation='false',MSBUILDDISABLENODEREUSE='1',DOTNET_CLI_DO_NOT_USE_MSBUILD_SERVER='1')
    def launch(name,command,cwd,env):
        record=dict(name=name,command=command,started=time.time(),members={},complete=False,code=None,samples=0);state['runs'].append(record);save(state_path,state)
        child=None;start=time.monotonic()
        try:
            with (base/(name+'.stdout')).open('x') as stdout,(base/(name+'.stderr')).open('x') as stderr,(base/(name+'.samples.jsonl')).open('x') as samples:
                child=subprocess.Popen(command,cwd=cwd,env=env,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
                process=ps.Process(child.pid);identity=dict(pid=process.pid,birth=process.create_time());record['child']=identity;record['members'][str(process.pid)]=identity['birth'];save(state_path,state)
                while child.poll() is None:
                    members=[]
                    try:
                        assert process.create_time()==identity['birth']
                        for member in [process]+process.children(recursive=True):
                            try:
                                birth=member.create_time();key=str(member.pid)
                                if key in record['members']:assert record['members'][key]==birth
                                record['members'][key]=birth
                                members.append(dict(pid=member.pid,birth=birth,rss=member.memory_info().rss,affinity=member.cpu_affinity()))
                            except ps.NoSuchProcess:pass
                    except ps.NoSuchProcess:pass
                    row=dict(seconds=time.monotonic()-start,available=ps.virtual_memory().available,members=members)
                    samples.write(json.dumps(row)+'\n');samples.flush();record['samples']+=1
                    assert row['seconds']<300 and sum(m['rss'] for m in members)<4*1024**3 and row['available']>=1024**3
                    assert all(m['affinity']==[0] for m in members)
                    time.sleep(.25)
                record['code']=child.returncode;assert child.returncode==0,(name,child.returncode)
                # With shared compilation and node reuse disabled, owned descendants exit.
                until=time.monotonic()+10
                while not all(absent(dict(pid=int(pid),birth=birth)) for pid,birth in record['members'].items()):
                    assert time.monotonic()<until,('Owned descendant remains',name);time.sleep(.1)
        except BaseException:
            record['error']=traceback.format_exc()
            for pid,birth in reversed(list(record['members'].items())):
                try:
                    member=ps.Process(int(pid))
                    if member.create_time()==birth:member.kill()
                except ps.NoSuchProcess:pass
            if child is not None:child.wait(timeout=10)
            raise
        finally:
            record['ended']=time.time();record['seconds']=time.monotonic()-start;record['complete']=True
            if child is not None:record['code']=child.poll()
            save(state_path,state)
        print(json.dumps(dict(stage=name,code=0,seconds=record['seconds'])),flush=True)
    try:
        buildenv=clean|dict(NUGET_PACKAGES=str(base/'build-packages'))
        flags=['--tl:off','--nologo','-v','minimal','-c','Release']
        launch('solution-build',['dotnet','build','Lokad.Onnx.slnx',*flags,'--disable-build-servers','-m:1'],base/'source',buildenv)
        launch('tensor-tests',['dotnet','test','tests/Lokad.Onnx.Tensors.Tests/Lokad.Onnx.Tensors.Tests.csproj',*flags,'--no-build','--no-restore','--logger','trx;LogFileName=tensors.trx','--results-directory',str(base/'test-results')],base/'source',buildenv)
        launch('backend-tests',['dotnet','test','tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj',*flags,'--no-build','--no-restore','--logger','trx;LogFileName=backend.trx','--results-directory',str(base/'test-results')],base/'source',buildenv)
        launch('pack',['dotnet','pack','src/Lokad.Onnx/Lokad.Onnx.csproj',*flags,'--no-build','--no-restore'],base/'source',buildenv)
        packages=list((base/'source/artifacts/nuget').glob('*.nupkg'));assert len(packages)==1
        shutil.copyfile(packages[0],base/'feed'/packages[0].name)
        app=base/'consumer';env=clean|dict(NUGET_PACKAGES=str(base/'packages'))
        launch('restore',['dotnet','restore','Consumer.csproj','--configfile','nuget.config','--tl:off','--nologo','-v','minimal'],app,env)
        launch('build',['dotnet','build','Consumer.csproj','-c','Release','--no-restore','--tl:off','--nologo','-v','minimal'],app,env)
        dll=app/'bin/Release/net10.0/LayerNormPackageConsumer.dll';fixture=base/'source/tests/Lokad.Onnx.Backend.Tests/models/mnist-8.onnx'
        for setting in SETTINGS:
            options={}
            if setting['fingerprint']:options['LOKAD_ONNX_FINGERPRINT_STRINGS']='1'
            if setting['wide']:options['LOKAD_ONNX_LAYERNORM_WIDE_OUTPUT']='1'
            command=['dotnet',str(dll),str(fixture),str(base/(setting['name']+'.json')),str(int(setting['fingerprint'])),str(int(setting['wide']))]
            launch(setting['name'],command,app,env|options)
        state['code']=0
    except BaseException:state['code']=1;state['error']=traceback.format_exc();raise
    finally:state['complete']=True;state['ended']=time.time();parent.cpu_affinity(old_affinity);save(state_path,state)

if __name__=='__main__':main()
