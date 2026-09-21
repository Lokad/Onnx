"""Run the four declared probe workers sequentially, then independently audit."""
import subprocess
import traceback
from common import *


def main():
    assert not (BASE/'finish-state.json').exists()
    capture=read(BASE/'capture-closed.json');assert capture['passed'];verify(capture['files'])
    files={p.relative_to(ROOT).as_posix():pin(p) for p in TOOLS.iterdir() if p.suffix in ('.py','.cs','.csproj')}
    save(BASE/'driver-prepared.json',dict(passed=True,files=files,capture=pin(BASE/'capture-closed.json')))
    own=psutil.Process();state=dict(complete=False,code=None,supervisor=dict(pid=own.pid,birth=own.create_time()),stages=[])
    try:
        for label,args in [(f'probe-{i}',[str(TOOLS/'run.py'),'probe',str(i)]) for i in range(4)]+[('audit',[str(TOOLS/'audit.py')])]:
            state['phase']=label;save(BASE/'finish-state.json',state)
            print('Starting '+label,flush=True)
            with (BASE/(label+'-supervisor.log')).open('x') as log:
                result=subprocess.run([sys.executable,'-X','utf8','-B',*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            state['stages'].append(dict(label=label,code=result.returncode));save(BASE/'finish-state.json',state)
            assert result.returncode==0,label
            print('Finished '+label,flush=True)
        state.update(code=0,passed=True)
    except BaseException:
        state.update(code=1,error=traceback.format_exc());raise
    finally:state['complete']=True;save(BASE/'finish-state.json',state)
    print(json.dumps(state))


if __name__=='__main__':main()
