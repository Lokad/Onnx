"""Wait on existing process identities, then collect and verify exactly once."""
import datetime,json,os,subprocess,time,traceback
from common import BASE,LOCAL_SITE,pin,read,write,sys,Path

MONITOR=BASE.with_name('audio-whisper-amd-finish-20260921')

def save(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8');temporary.replace(path)

def main():
    folder=Path(__file__).parent.resolve();state_path=MONITOR/'state.json'
    assert not MONITOR.exists() and not (BASE/'collected').exists()
    assert (BASE/'deployment.json').exists()
    MONITOR.mkdir()
    sys.path.insert(0,str(LOCAL_SITE));import psutil
    own=psutil.Process();own.cpu_affinity([0])
    tools={name:pin(folder/name) for name in ['observe.py','collect.py','audit.py','close.py','verify.py','common.py']}
    state=dict(complete=False,code=None,supervisor=dict(pid=own.pid,birth=own.create_time()),deployment=read(BASE/'deployment.json'),
        tools=tools,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),observations=0,stages=[])
    write(state_path,state);start=time.monotonic();env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1')
    child=None
    try:
        with (MONITOR/'observations.jsonl').open('x',encoding='utf-8') as observations:
            while True:
                assert time.monotonic()-start<14400,'Completion monitor time limit; inspect the original remote job before any action'
                assert pin(folder/'observe.py')==tools['observe.py'] and pin(folder/'common.py')==tools['common.py']
                try:
                    result=subprocess.run([sys.executable,'-X','utf8','-B',str(folder/'observe.py')],env=env,capture_output=True,text=True,timeout=60,creationflags=subprocess.CREATE_NO_WINDOW)
                    observation=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),code=result.returncode,stdout=result.stdout,stderr=result.stderr)
                except subprocess.TimeoutExpired as error:
                    observation=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),code=None,timeout=str(error))
                    observations.write(json.dumps(observation)+'\n');observations.flush();state['observations']+=1
                    state['last_observation_error']=observation;save(state_path,state);time.sleep(30);continue
                observations.write(json.dumps(observation)+'\n');observations.flush();state['observations']+=1
                if result.returncode==0:
                    value=json.loads(result.stdout);state['last_observation']=value;save(state_path,state)
                    if value.get('complete') and not value['supervisor_live'] and all(not b['live'] for r in value['runs'] for b in r['births']):
                        terminal=value;break
                else:
                    state['last_observation_error']=observation;save(state_path,state)
                # A failed observation never restarts inference or authorizes collection.
                time.sleep(30)
        names=['collect.py']+(['audit.py','close.py','verify.py'] if terminal['code']==0 else [])
        for name in names:
            assert pin(folder/name)==tools[name],name
            stage=dict(name=name,complete=False,code=None,started=time.time());state['stages'].append(stage);save(state_path,state)
            with (MONITOR/(name+'.stdout')).open('x') as out,(MONITOR/(name+'.stderr')).open('x') as err:
                child=subprocess.Popen([sys.executable,'-X','utf8','-B',str(folder/name)],env=env,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
                stage['child']=dict(pid=child.pid,birth=psutil.Process(child.pid).create_time());save(state_path,state)
                stage['code']=child.wait(timeout=600);stage.update(complete=True,ended=time.time());save(state_path,state)
                assert stage['code']==0,(name,stage['code'])
                child=None
            print(json.dumps(dict(stage=name,code=0)),flush=True)
        assert terminal['code']==0,'Failed remote campaign collected; success reporting deliberately not run'
        assert read(BASE/'final-verification.json')['passed'];state['code']=0
    except BaseException:
        state.update(code=1,error=traceback.format_exc())
        if child is not None and child.poll() is None:
            # Terminate only the recorded local reporting process and its actual children.
            try:
                identity=state['stages'][-1]['child'];p=psutil.Process(identity['pid'])
                if p.create_time()==identity['birth']:
                    descendants=p.children(recursive=True)
                    for member in reversed(descendants):member.kill()
                    p.kill()
            except psutil.NoSuchProcess:pass
            child.wait(timeout=10)
        raise
    finally:
        state.update(complete=True,ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.monotonic()-start);save(state_path,state)

if __name__=='__main__':main()
