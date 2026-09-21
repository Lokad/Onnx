"""Stage and launch only after the complete e5 controller and remote births close."""
import shutil
from common import *
from storage_contract import safe_member


def e5_gate():
    state = read(E5_CONTROL/'state.json')
    assert state['complete'] is True and state['code'] == 0 and absent(state['supervisor']), 'e5 controller still active or failed'
    assert all(s['complete'] and s['code'] == 0 and absent(s['child']) for s in state['stages'])
    phase = state['phase']; assert phase in ['aa','compare']
    verified = read(E5/(phase+'-verification.json')); assert verified['passed']
    receipt = E5/('collected-'+phase)/(phase+'-collection.json')
    inventory = read(receipt)
    assert inventory['terminal'] and inventory['code'] == 0 and inventory['remote_audit_code'] == 0
    assert pin(E5/'frozen.json') == verified['frozen'] == inventory['frozen']
    for field, name in [('report',phase+'-results.md'),('observations',phase+'-observations.json')]:
        assert pin(ROOT/'tests/e5/randomized-processes'/name) == verified[field]
    # All phases must be fully closed, including the comparison when it ran.
    births = []; phases = ['aa','compare'] if phase == 'compare' else ['aa']
    for p in phases:
        proof = read(E5/(p+'-verification.json')); assert proof['passed']
        collection = read(E5/('collected-'+p)/(p+'-collection.json'))
        assert collection['terminal'] and collection['code'] == 0 and collection['remote_audit_code'] == 0
        births.extend(collection['births'])
    return dict(passed=True, phase=phase, controller=pin(E5_CONTROL/'state.json'), verification=pin(E5/(phase+'-verification.json')),
        collection=pin(receipt), remote_receipt=phase+'-collection.json', births=births)


def main():
    prepared = read(BASE/'prepared.json'); assert prepared['passed']
    for name, expected in prepared['sources'].items():
        assert pin(ROOT/name) == expected, name
    assert pin(BASE/'frozen-template.json') == prepared['template']
    gate = e5_gate()  # No SSH, staging or VM mutation before this barrier passes.
    assert not (BASE/'stage.json').exists() and not (BASE/'deployment.json').exists()
    for name in prepared['copies'] | prepared['uploads']:
        safe_member(name)
    frozen = read(BASE/'frozen-template.json'); frozen['e5_gate'] = gate
    frozen['files']['e5-gate.json'] = None
    write(BASE/'e5-gate.json', gate); frozen['files']['e5-gate.json'] = pin(BASE/'e5-gate.json')
    script = PRELUDE+'''
terminal(%r)
assert pin(e5/%r)==%r
assert not base.exists() and base.parent==Path('/dev/shm') and base.parent.resolve()==base.parent
assert psutil.virtual_memory().available>=13*1024**3 and shutil.disk_usage(base.parent).free>=3*1024**3
assert os.stat('/dev/shm').st_dev!=os.stat('/').st_dev
for name,wanted in %r.items():assert pin(old/name)==wanted,name
for name,wanted in %r.items():assert pin(name)==wanted,name
base.mkdir()
for name,wanted in %r.items():
 target=base/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(old/name,target);assert pin(target)==wanted
for name in %r:(base/name).parent.mkdir(parents=True,exist_ok=True)
print(json.dumps(dict(staged=True,files=%r,root_free=shutil.disk_usage('/').free,output_free=shutil.disk_usage(base).free)))
''' % (gate['births'], gate['remote_receipt'], gate['collection'], prepared['copies'], frozen['external'], prepared['copies'], list(prepared['uploads']), len(prepared['copies']))
    write(BASE/'stage.json', json.loads(ssh(script,timeout=1800)))
    uploads = {n:BASE/'payload'/n for n in prepared['uploads']}; uploads['e5-gate.json'] = BASE/'e5-gate.json'
    for name, path in uploads.items():
        assert pin(path) == frozen['files'][name]
        subprocess.run(['scp','-i',KEY,'-o','BatchMode=yes',str(path),HOST+':'+REMOTE+'/'+name],check=True,
            timeout=120,creationflags=subprocess.CREATE_NO_WINDOW)
    script = PRELUDE+'''
terminal(%r);frozen=%r
sys.path.insert(0,str(base/'runtime'))
from storage_contract import observe_storage,validate_storage
validate_storage(observe_storage(base),preflight=True)
for name,wanted in frozen['files'].items():assert pin(base/name)==wanted,name
for name,wanted in frozen['external'].items():assert pin(name)==wanted,name
assert {p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file()}==set(frozen['files'])
write(base/'frozen.json',frozen)
print(json.dumps(dict(frozen=pin(base/'frozen.json'),storage=observe_storage(base))))
''' % (gate['births'], frozen)
    result = json.loads(ssh(script,timeout=1800)); write(BASE/'freeze-receipt.json',result)
    # Preserve the Linux serialization bytes; Windows text writing translates newlines.
    subprocess.run(['scp','-i',KEY,'-o','BatchMode=yes',HOST+':'+REMOTE+'/frozen.json',str(BASE/'frozen.json')],check=True,
        timeout=120,creationflags=subprocess.CREATE_NO_WINDOW)
    assert pin(BASE/'frozen.json') == result['frozen']
    script = PRELUDE+'''
terminal(%r);assert pin(base/'frozen.json')==%r
sys.path.insert(0,str(base/'runtime'))
from storage_contract import observe_storage,validate_storage
validate_storage(observe_storage(base),preflight=True)
assert not (base/'deployment.json').exists() and not (base/'campaign').exists()
assert psutil.virtual_memory().available>=13*1024**3
frozen=read(base/'frozen.json')
env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
env.update(PYTHONPATH=os.pathsep.join(frozen['python_paths']),PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1')
env.update({k:'1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS']})
with (base/'supervisor.stdout').open('x') as out,(base/'supervisor.stderr').open('x') as err:
 p=subprocess.Popen(['python3','-B',str(base/'runtime/supervise.py'),str(base)],cwd=base,env=env,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True)
result=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),frozen=pin(base/'frozen.json'))
write(base/'deployment.json',result);print(json.dumps(result))
''' % (gate['births'],result['frozen'])
    deployment = json.loads(ssh(script)); write(BASE/'deployment.json',deployment)
    print(json.dumps(deployment))


if __name__ == '__main__':
    main()
