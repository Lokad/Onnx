"""Start the frozen comparison only after complete independently verified A/A."""
import json

from contract import pin, read, write
from remote import BASE, REMOTE, PRELUDE, ssh, copy_to


def main():
    assert not (BASE/'deployment-compare.json').exists()
    verification = read(BASE/'aa-verification.json')
    assert verification['passed'] is True and verification['statistical_screen'] is True and verification['diagnostic_screen'] is True
    gate_path = BASE/'aa-gate.json'; gate = read(gate_path)
    assert gate['passed'] is True and gate['independent_verification'] == pin(BASE/'aa-verification.json')
    assert gate['frozen'] == pin(BASE/'frozen.json') == verification['frozen']
    script = PRELUDE+'''
os.sched_setaffinity(0,{0});terminal(%r)
assert pin(base/'frozen.json')==%r
assert not (base/'aa-gate.json').exists() and not (base/'deployment-compare.json').exists() and not (base/'result-compare').exists()
assert psutil.virtual_memory().available>=10*1024**3 and shutil.disk_usage(base).free>=3*1024**3
for name,wanted in %r.items():assert pin(base/name)==wanted,name
print(json.dumps(dict(ready=True)))
''' % (gate['resources']['births'], gate['frozen'], gate['retained_files'])
    assert json.loads(ssh(script))['ready']
    copy_to(gate_path, REMOTE+'/aa-gate.json')
    script = PRELUDE+'''
os.sched_setaffinity(0,{0});terminal(%r)
assert pin(base/'aa-gate.json')==%r and pin(base/'frozen.json')==%r
assert not (base/'deployment-compare.json').exists() and not (base/'result-compare').exists()
meta=read(base/'frozen.json')
env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
env.update(PYTHONPATH=os.pathsep.join(meta['python_paths']),PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1')
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='1'
with (base/'supervisor-compare.stdout').open('x') as out,(base/'supervisor-compare.stderr').open('x') as err:
 p=subprocess.Popen([meta['python_executable'],'-B',str(base/'tests/e5/randomized-processes/run.py'),'--payload',str(base),'--phase','compare','--gate-sha256',%r],cwd=base,env=env,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True)
deployment=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),frozen=pin(base/'frozen.json'),gate=pin(base/'aa-gate.json'))
write(base/'deployment-compare.json',deployment);print(json.dumps(deployment))
''' % (gate['resources']['births'], pin(gate_path), gate['frozen'], pin(gate_path)['sha256'])
    deployment = json.loads(ssh(script)); write(BASE/'deployment-compare.json', deployment)
    print(json.dumps(deployment))


if __name__ == '__main__':
    main()
