"""Exercise the new runner with the already qualified unchanged worker binary."""
from pathlib import Path
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'artifacts/e5-randomized-processes-20260921'
SITE = ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'
sys.path.append(str(SITE))
import psutil
from contract import pin, read, write, manifest, audit, CORE, LIMITS, CRITERIA
from design import PROTOCOL, assignment_schedule
from run import terminal, save


def main():
    base = BASE/'local-payload'
    assert not base.exists()
    assignments = BASE/'assignments-frozen.json'
    assert pin(assignments) == dict(bytes=889069, sha256='2ac2bd42acfc811b39f2df099c7e93f31789bea18236e0e507d74e60ba5b16f5')
    prior = ROOT/'artifacts/e5-process-uncertainty-20260921'
    qualification = read(prior/'local-verification.json')
    assert qualification['passed'] is True
    for name, want in qualification['pins'].items():
        assert pin(prior/name) == want, name
    base.mkdir(); shutil.copytree(prior/'bin', base/'bin'); shutil.copytree(prior/'inputs', base/'inputs')
    shutil.copyfile(assignments, base/'assignments-frozen.json')
    for folder_name, names in [
        ('randomized-processes', ['design.py', 'contract.py', 'run.py']),
        ('process-uncertainty', ['estimator.py', 'protocol.py', 'worker_audit.py'])]:
        destination = base/'tests/e5'/folder_name; destination.mkdir(parents=True)
        for name in names:
            shutil.copyfile(ROOT/'tests/e5'/folder_name/name, destination/name)
    dotnet = Path(shutil.which('dotnet')).resolve()
    runtime = dotnet.parent/'shared/Microsoft.NETCore.App/10.0.12'
    assert runtime.is_dir()
    def external(path):
        return dict(path=path.as_posix(), **pin(path))
    chosen = read(assignments)
    schedules = {}
    for phase in ('aa', 'compare'):
        jobs = assignment_schedule(chosen['draws'][phase], phase)
        schedules[phase] = [j for j in jobs if j['cohort'] == 0 and (j['case_index'], j['policy']) in ((0, 'default'), (4, 'memory'))]
    meta = dict(protocol=PROTOCOL, worker_protocol='e5-fresh-process-uncertainty-v1', mode='smoke',
                core_sha256=CORE, limits=LIMITS, criteria=CRITERIA, schedules=schedules,
                model=external(ROOT/'models/multilingual-e5-small/model.onnx'),
                native=external(ROOT/'artifacts/e5-public-ort-20260919/bin/onnxruntime.dll'), dotnet=external(dotnet),
                runtime_files=[external(p) for p in sorted(runtime.rglob('*')) if p.is_file()],
                producer_qualification=pin(prior/'local-verification.json'),
                files={p.relative_to(base).as_posix(): pin(p) for p in sorted(base.rglob('*')) if p.is_file()})
    write(base/'frozen.json', meta)
    own = psutil.Process(); original_affinity = own.cpu_affinity(); own.cpu_affinity([0])
    state = dict(supervisor=dict(pid=own.pid, birth=own.create_time()), started=time.time(), complete=False,
                 frozen=pin(base/'frozen.json'), children=[], phases=[])
    save(BASE/'local-state.json', state)
    env = dict(os.environ, PYTHONPATH=str(SITE), PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1')
    child = None
    try:
        for phase in ('aa', 'compare'):
            with (BASE/('local-'+phase+'.stdout')).open('x') as stdout, (BASE/('local-'+phase+'.stderr')).open('x') as stderr:
                child = subprocess.Popen([sys.executable, '-X', 'utf8', '-B', str(base/'tests/e5/randomized-processes/run.py'),
                                          '--payload', str(base), '--phase', phase], env=env, stdout=stdout, stderr=stderr,
                                         creationflags=subprocess.CREATE_NO_WINDOW)
                state['children'].append(dict(pid=child.pid, birth=psutil.Process(child.pid).create_time()))
                save(BASE/'local-state.json', state)
                code = child.wait(timeout=300); assert code == 0, (phase, code)
                child = None
            terminal(state['children'])
            result = audit(base, phase)
            terminal(result['resources']['births'])
            write(BASE/('local-'+phase+'-audit.json'), result)
            state['phases'].append(dict(phase=phase, audit=pin(BASE/('local-'+phase+'-audit.json')))); save(BASE/'local-state.json', state)
            print(json.dumps(dict(phase=phase, measured=result['measured_calls'], conditioning=result['conditioning_calls'], passed=True)), flush=True)
        mutations = [dict(protocol='old'), dict(worker_protocol='wrong'), dict(mode='unknown'), dict(core_sha256='0'*64),
                     dict(limits={}), dict(criteria={}), dict(schedules={'aa': [], 'compare': []})]
        refusals = 0
        with tempfile.TemporaryDirectory(prefix='manifest-refusals-', dir=BASE) as temporary:
            temporary = Path(temporary)
            # Hardlink immutable inputs only. The manifest itself is a separate
            # file, so deliberate mutations cannot change the qualified payload.
            for name in meta['files']:
                destination = temporary/name; destination.parent.mkdir(parents=True, exist_ok=True)
                os.link(base/name, destination)
            for changes in mutations:
                (temporary/'frozen.json').write_text(json.dumps(meta | changes), encoding='utf8')
                try:
                    manifest(temporary)
                except (AssertionError, KeyError, ValueError):
                    refusals += 1
                else:
                    raise AssertionError(('bad manifest accepted', changes))
            for file in ('bin/ProcessUncertainty.dll', 'bin/Lokad.Onnx.dll', 'assignments-frozen.json'):
                changed = copy.deepcopy(meta); changed['files'][file]['sha256'] = '0'*64
                (temporary/'frozen.json').write_text(json.dumps(changed), encoding='utf8')
                try:
                    manifest(temporary)
                except AssertionError:
                    refusals += 1
                else:
                    raise AssertionError(('bad identity accepted', file))
        assert refusals == 10
        manifest(base)  # All deliberate refusals left original bytes intact.
        state.update(code=0, complete=True, manifest_refusals=refusals)
    except BaseException:
        state.update(code=2, error=traceback.format_exc())
        if child is not None and child.poll() is None:
            identity = state['children'][-1]
            try:
                process = psutil.Process(identity['pid'])
                if process.create_time() == identity['birth']:
                    for member in reversed(process.children(recursive=True)):
                        member.kill()
                    process.kill()
            except psutil.NoSuchProcess:
                pass
            child.wait(timeout=10)
        raise
    finally:
        state['ended'] = time.time(); save(BASE/'local-state.json', state); own.cpu_affinity(original_affinity)
    print(json.dumps(dict(passed=True, workers=12, manifest_refusals=10, frozen=pin(base/'frozen.json'))))


if __name__ == '__main__':
    main()
