"""Run only the two M78 observation processes refused before launch."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from rules import verify_refusal


def adapted_capture(source):
    tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='capture')
    body=ast.get_source_segment(source,node)
    before="for mode in ['control','phase','wall']:"
    after="for mode in ['phase','wall']:\n        wait_for_headroom(state,mode,spec)"
    assert body.count(before)==1
    changed=body.replace(before,after)
    assert changed.replace(after,before)==body
    return changed


def verify():
    global original
    import remote as original
    base=original.BASE;details=original.read(base/'resume-spec.json')
    for name,wanted in details['files'].items():assert original.pin(base/name)==wanted,name
    assert original.pin(base/'spec.json')==details['original_spec']
    assert original.pin(base/'remote.py')==details['original_worker']
    assert original.pin(base/'capture-state.json')==details['original_state']
    assert original.pin(base/'capture-collection.json')==details['original_collection']
    assert original.pin(base/'build-review.json')==details['build_review']
    assert original.pin(base/'built.json')==details['built']
    for name,wanted in details['retained_files'].items():assert original.pin(base/name)==wanted,name
    spec=original.verify();state=original.read(base/'capture-state.json')
    assert verify_refusal(state,spec)==details['refusal']
    owners=[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    assert not any(original.live(i) for i in owners)
    return spec,details


def wait_for_headroom(state,mode,spec):
    original.idle();start=time.monotonic()
    record=dict(mode=mode,observations=[]);state['headroom_waits'].append(record)
    while True:
        row=dict(seconds=time.monotonic()-start,available=original.psutil.virtual_memory().available,
                 tmpfs=original.psutil.disk_usage(original.BASE).free)
        record['observations'].append(row);original.save(original.BASE/'resume-state.json',state)
        assert row['seconds']<900,'Pre-launch headroom did not recover within 900 seconds'
        if row['available']>=spec['capture_limits']['available_before']+64*1024**2 and row['tmpfs']>=spec['capture_limits']['tmpfs_before']:
            return
        time.sleep(1)


def main():
    spec,details=verify();original.idle();base=original.BASE
    assert not (base/'resume-state.json').exists()
    for mode in ['phase','wall']:
        assert not (base/mode).exists()
        assert not any((base/'logs').glob(mode+'.*'))
    own=original.psutil.Process();own.cpu_affinity([0])
    state=dict(kind='resume',complete=False,code=None,supervisor=dict(pid=own.pid,birth=own.create_time()),
               runs=[],headroom_waits=[],started=time.time(),original_state=details['original_state'])
    original.save(base/'resume-state.json',state)
    env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_','parakeet_phase_'))}
    env.pop('PYTHONOPTIMIZE',None)
    env['PATH']=str(Path(original.DOTNET).parent)+os.pathsep+env.get('PATH','')
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[name]='1'
    source=adapted_capture((base/'remote.py').read_text())
    assert hashlib.sha256(source.encode()).hexdigest()==details['capture_source_sha256']
    namespace=dict(vars(original),wait_for_headroom=wait_for_headroom)
    exec(compile(source,'original-capture-unstarted-modes-only','exec'),namespace)
    try:
        namespace['capture'](state,env,spec)
        verify();state['code']=0
    except BaseException:
        state.update(code=1,error=traceback.format_exc());traceback.print_exc()
    finally:
        state.update(complete=True,ended=time.time());original.save(base/'resume-state.json',state)
    assert state['code']==0


if __name__=='__main__':main()
