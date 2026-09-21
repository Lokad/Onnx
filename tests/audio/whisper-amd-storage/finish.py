"""Wait for the actual e5 controller, then execute the storage-corrected lane once."""
import time
import traceback
from common import *

CONTROL = BASE.with_name('audio-whisper-storage-finish-20260921')


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf8'); temporary.replace(path)


def main():
    import psutil
    assert not CONTROL.exists() and not (BASE/'deployment.json').exists()
    prepared = pin(BASE/'prepared.json'); spec = read(BASE/'prepared.json'); assert spec['passed']
    own = psutil.Process(); own.cpu_affinity([0]); CONTROL.mkdir()
    state = dict(complete=False,code=None,phase='waiting-for-e5',started=time.time(),
        supervisor=dict(pid=own.pid,birth=own.create_time()),prepared=prepared,stages=[],observations=0)
    state_path = CONTROL/'state.json'; save(state_path,state)
    child = None
    env = dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1')

    def unchanged():
        assert pin(BASE/'prepared.json') == prepared
        for name, expected in spec['sources'].items():
            assert pin(ROOT/name) == expected, name

    def step(name):
        nonlocal child
        unchanged(); stage=dict(name=name,complete=False,code=None,started=time.time())
        state['stages'].append(stage);save(state_path,state)
        with (CONTROL/(name+'.stdout')).open('x') as out,(CONTROL/(name+'.stderr')).open('x') as err:
            child=subprocess.Popen([sys.executable,'-X','utf8','-B',str(TOOLS/name)],cwd=ROOT,env=env,stdout=out,stderr=err,
                creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW)
            stage['child']=dict(pid=child.pid,birth=psutil.Process(child.pid).create_time());save(state_path,state)
            stage['code']=child.wait(timeout=7200)
        stage.update(complete=True,ended=time.time());save(state_path,state)
        assert stage['code']==0,(name,stage['code'])
        assert absent(stage['child']);child=None
        print(json.dumps(dict(stage=name,code=0)),flush=True)

    try:
        started=time.monotonic()
        while True:
            assert time.monotonic()-started < 72*3600, 'Wait ceiling; inspect original e5 identities'
            unchanged(); previous=read(E5_CONTROL/'state.json')
            live=not absent(previous['supervisor']);state['predecessor']=dict(complete=previous['complete'],code=previous['code'],live=live,identity=previous['supervisor'])
            save(state_path,state)
            if previous['complete']:
                assert previous['code']==0,'e5 failed: no Whisper deployment'
                if not live:
                    break
            else:
                assert live,'e5 controller disappeared: inspect, do not launch'
            time.sleep(30)
        state['phase']='stage';save(state_path,state);step('stage.py')
        state['phase']='inference';save(state_path,state);started=time.monotonic()
        with (CONTROL/'observations.jsonl').open('x',encoding='utf8') as log:
            while True:
                assert time.monotonic()-started < 5*3600,'Observation ceiling: inspect existing VM births'
                unchanged()
                try:
                    result=subprocess.run([sys.executable,'-X','utf8','-B',str(TOOLS/'observe.py')],cwd=ROOT,env=env,
                        capture_output=True,text=True,encoding='utf8',timeout=75,creationflags=subprocess.CREATE_NO_WINDOW)
                    event=dict(time=time.time(),code=result.returncode,stdout=result.stdout,stderr=result.stderr)
                except subprocess.TimeoutExpired:
                    event=dict(time=time.time(),code=None,timeout=True)
                log.write(json.dumps(event)+'\n');log.flush();state['observations']+=1
                if event['code']==0:
                    value=json.loads(event['stdout']);state['last_observation']=value
                    if not value['supervisor_live']:
                        assert not value.get('initializing'),'Supervisor stopped before state creation; retain stderr'
                        if all(not b['live'] for r in value['runs'] for b in r['births']):
                            assert value['complete'] is True
                            break
                else:
                    state['last_observation_error']=event
                save(state_path,state);time.sleep(30)
        state['phase']='collection';save(state_path,state);step('collect.py')
        assert value['code']==0,'Failed inference collected; no success report or retry'
        for name in ['audit.py','close.py','verify.py']:
            state['phase']=name;save(state_path,state);step(name)
        assert read(BASE/'final-verification.json')['passed']
        state.update(code=0,outcome='Four fresh timing workers fully collected, audited and independently reported')
    except BaseException:
        state.update(code=1,error=traceback.format_exc())
        if child is not None:
            state['last_child_still_live']=child.poll() is None
        raise
    finally:
        state.update(complete=True,ended=time.time());save(state_path,state)


if __name__=='__main__':
    main()
