"""Own the fixed worker sequence and its final audit without automatic retries."""
import json
import subprocess
import time
import traceback
from common import *


def main():
    assert pin(BASE/'prepared.json')['sha256']=='05a5fb11e3d2e80daf2955467e1745a43d8839da6b6c72d66ee9dfeb35ea50e0'
    assert not (BASE/'finish-state.json').exists()
    own=psutil.Process();prior=own.cpu_affinity();own.cpu_affinity([0])
    state=dict(complete=False,code=None,supervisor=dict(pid=own.pid,birth=own.create_time()),stages=[],
        tools={name:pin(TOOLS/name) for name in ('run.py','audit.py','finish.py')},prepared=pin(BASE/'prepared.json'))
    save(BASE/'finish-state.json',state)
    try:
        for phase in ('run','audit'):
            row=dict(phase=phase,complete=False,code=None);state['stages'].append(row)
            with (BASE/(phase+'-supervisor.log')).open('x') as log:
                child=subprocess.Popen([sys.executable,'-X','utf8','-B',str(TOOLS/(phase+'.py'))],cwd=ROOT,
                    stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                    creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NO_WINDOW)
                process=psutil.Process(child.pid);row['worker']=dict(pid=child.pid,birth=process.create_time());save(BASE/'finish-state.json',state)
                row['code']=child.wait();row['complete']=True;save(BASE/'finish-state.json',state);assert row['code']==0,(phase,row['code'])
            terminal(row['worker'])
        state['code']=0
    except BaseException:state.update(code=1,error=traceback.format_exc());raise
    finally:state['complete']=True;save(BASE/'finish-state.json',state);own.cpu_affinity(prior)
    print(json.dumps(dict(passed=True,state=pin(BASE/'finish-state.json'),closed=pin(BASE/'closed.json'))))


if __name__=='__main__':main()
