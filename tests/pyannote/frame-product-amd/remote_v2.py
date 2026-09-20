"""Finite Linux source-archive qualification of the public WeSpeaker frontend."""
from pathlib import Path
import hashlib, json, os, subprocess, sys, time, shutil, traceback

sys.path.insert(0, '/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def absent(identity):
    try:
        return psutil.Process(identity['pid']).create_time() != identity['birth']
    except psutil.NoSuchProcess:
        return True


def check(row):
    assert 0 <= row['seconds'] < 600
    assert row['available'] >= 1024**3 and row['disk'] >= 1024**3
    assert sum(m['rss'] for m in row['members']) < 2*1024**3
    assert all(m['affinity'] == [2] for m in row['members'])


def run(base):
    assert psutil.__version__ == '7.0.0'
    payload = json.loads((base/'payload.json').read_text())
    for name, value in payload['files'].items():
        assert pin(base/name) == value, name
    own = psutil.Process(); own.cpu_affinity([0])
    result = base/'result'; result.mkdir()
    state = dict(source=payload['source'], tools=payload['tools'], payload=pin(base/'payload.json'),
                 supervisor=dict(pid=own.pid, birth=own.create_time()), complete=False, code=None, runs=[])
    def save():
        temp = result/'run.tmp'; temp.write_text(json.dumps(state, indent=2)); temp.replace(result/'run.json')
    source = base/'source'; binary = source/'tests/Lokad.Onnx.Backend.Tests/bin/Release/net10.0'
    env = {k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_', 'dotnet_', 'complus_'))}
    env.update(DOTNET_PROCESSOR_COUNT='1', UseSharedCompilation='false', MSBUILDDISABLENODEREUSE='1',
               DOTNET_CLI_DO_NOT_USE_MSBUILD_SERVER='1', DOTNET_CLI_TELEMETRY_OPTOUT='1')
    common = ['--tl:off','--nologo','-v','minimal','-p:UseSharedCompilation=false','-p:BuildInParallel=false','-m:1']
    commands = [
        ('cli-build',['dotnet','build','src/Lokad.Onnx.CLI','-c','Release',*common,'--disable-build-servers']),
        ('tests', ['dotnet','test','tests/Lokad.Onnx.Backend.Tests','-c','Release',*common,
                   '--filter','FullyQualifiedName~WeSpeaker|FullyQualifiedName~Community1',
                   '--logger','trx;LogFileName=affected.trx','--results-directory',str(result/'trx')]),
        ('consumer-build',['dotnet','build','tests/pyannote/frame-product/Consumer.csproj','-c','Release',*common,
                           '--disable-build-servers','-o',str(result/'consumer')]),
        ('dc',['dotnet',str(result/'consumer/Consumer.dll'),str(binary),str(base/'reference'),str(result/'dc'),'dc-check']),
        ('corpus',['dotnet',str(result/'consumer/Consumer.dll'),str(binary),str(base/'reference'),str(result/'corpus'),'corpus'])]
    try:
        state['host'] = subprocess.check_output(['uname','-a'],text=True).strip()
        state['sdk'] = subprocess.check_output(['dotnet','--version'],cwd=source,env=env,text=True).strip()
        assert state['sdk']=='10.0.204'
        for name, command in commands:
            available=psutil.virtual_memory().available; disk=psutil.disk_usage(str(base)).free
            assert available>=4*1024**3 and disk>=2*1024**3
            record=dict(name=name, command=command, preflight_available=available, preflight_disk=disk,
                        started=time.time(), members={}, samples=0, complete=False, code=None)
            state['runs'].append(record); save(); child=None; start=time.monotonic()
            try:
                with (result/(name+'.stdout')).open('x') as stdout, (result/(name+'.stderr')).open('x') as stderr, (result/(name+'.samples.jsonl')).open('x') as log:
                    child=subprocess.Popen(['taskset','-c','2',*command],cwd=source,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
                    process=psutil.Process(child.pid); record['child']=dict(pid=child.pid,birth=process.create_time())
                    record['members'][str(child.pid)]=record['child']['birth']; save()
                    while child.poll() is None:
                        members=[]
                        try:
                            assert process.create_time()==record['child']['birth']
                            for p in [process]+process.children(recursive=True):
                                try:
                                    birth=p.create_time(); assert record['members'].get(str(p.pid),birth)==birth
                                    record['members'][str(p.pid)]=birth
                                    members.append(dict(pid=p.pid,birth=birth,rss=p.memory_info().rss,affinity=p.cpu_affinity()))
                                except psutil.NoSuchProcess: pass
                        except psutil.NoSuchProcess: pass
                        row=dict(seconds=time.monotonic()-start, available=psutil.virtual_memory().available,
                                 disk=psutil.disk_usage(str(base)).free,members=members)
                        log.write(json.dumps(row)+'\n'); log.flush(); record['samples']+=1; save(); check(row); time.sleep(.25)
                    record['code']=child.wait(); assert record['code']==0
                    deadline=time.monotonic()+10
                    while not all(absent(dict(pid=int(pid),birth=birth)) for pid,birth in record['members'].items()):
                        assert time.monotonic()<deadline; time.sleep(.1)
            except BaseException as error:
                record['error']=repr(error)
                for pid,birth in reversed(list(record['members'].items())):
                    try:
                        p=psutil.Process(int(pid))
                        if p.create_time()==birth: p.kill()
                    except psutil.NoSuchProcess: pass
                if child is not None: child.wait(timeout=10)
                raise
            finally:
                record.update(complete=True,seconds=time.monotonic()-start,ended=time.time()); save()
            print(json.dumps(dict(name=name,code=record['code'],seconds=record['seconds'])),flush=True)
        # Preserve the exact tested product/dependencies for the later connected replay.
        target=result/'product-bin';target.mkdir()
        for path in sorted(binary.iterdir()):
            if path.is_file() and (path.suffix in ['.dll','.json']): shutil.copyfile(path,target/path.name)
        for name,value in payload['files'].items(): assert pin(base/name)==value, name
        state['code']=0
    except BaseException as error:
        state.update(code=1,error=repr(error)); traceback.print_exc()
    finally:
        state['complete']=True;save()
    return state['code']


if __name__=='__main__':
    sys.exit(run(Path(sys.argv[1]).resolve()))
