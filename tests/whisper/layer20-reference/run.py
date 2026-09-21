"""Run the fixed 48 local references once, stopping on resource or pair failure."""
import json
import os
import shutil
import subprocess
import time
import traceback
from protocol import *


def save(path, value):
    temporary = path.with_suffix('.tmp'); temporary.write_text(json.dumps(value, indent=2), encoding='utf8'); temporary.replace(path)


def main():
    spec = read(BASE/'manifest.json'); assert spec['protocol'] == PROTOCOL and len(spec['jobs']) == 48
    assert spec['limits'] == LIMITS and spec['threads'] == THREADS
    for name, wanted in spec['files'].items():
        assert pin(ROOT/name) == wanted, name
    for name, wanted in spec['numerical_files'].items():
        assert pin(name) == wanted, name
    path = BASE/'processes.json'; assert not path.exists()
    parent = psutil.Process(); previous_affinity = parent.cpu_affinity(); parent.cpu_affinity([0])
    state = dict(complete=False, code=None, started=time.time(), manifest=pin(BASE/'manifest.json'),
                 supervisor=dict(pid=parent.pid, birth=parent.create_time()), runs=[], pairs=[])
    save(path, state)
    try:
        for job in spec['jobs']:
            assert psutil.virtual_memory().available >= LIMITS['preflight_available']
            assert shutil.disk_usage(BASE).free >= LIMITS['disk']
            folder = BASE/'process'/job['id']; folder.mkdir(parents=True, exist_ok=False)
            run = dict(job=job, complete=False, code=None, started=time.time(), samples=0, peak_rss=0)
            state['runs'].append(run); save(path, state); child = None; birth = None; started = time.monotonic()
            try:
                env = {k: v for k, v in os.environ.items() if not k.lower().startswith(('lokad_', 'dotnet_', 'complus_'))}
                env.update(THREADS); env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1')
                with (folder/'stdout.txt').open('x') as out, (folder/'stderr.txt').open('x') as err, (folder/'samples.jsonl').open('x') as samples:
                    parent.cpu_affinity([2])
                    try:
                        child = subprocess.Popen([sys.executable, '-X', 'utf8', '-B', str(Path(__file__).with_name('worker.py')), '--job', job['id']],
                                cwd=ROOT, env=env, stdout=out, stderr=err,
                                creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW)
                    finally:
                        parent.cpu_affinity([0])
                    process = psutil.Process(child.pid); birth = dict(pid=child.pid, birth=process.create_time()); run['worker'] = birth; save(path, state)
                    while child.poll() is None:
                        try:
                            assert process.create_time() == birth['birth'] and not process.children(recursive=True)
                            sample = dict(seconds=time.monotonic()-started, rss=process.memory_info().rss, available=psutil.virtual_memory().available,
                                          affinity=process.cpu_affinity(), pid=process.pid, birth=process.create_time())
                        except psutil.NoSuchProcess:
                            continue
                        samples.write(json.dumps(sample)+'\n'); samples.flush(); run['samples'] += 1; run['peak_rss'] = max(run['peak_rss'], sample['rss'])
                        assert sample['seconds'] < LIMITS['seconds'] and sample['rss'] < LIMITS['rss'] and sample['available'] >= LIMITS['available']
                        assert sample['affinity'] == [2]
                        time.sleep(.25)
                    run['code'] = child.wait(); assert run['code'] == 0, (job['id'], run['code'])
            except BaseException:
                run['error'] = traceback.format_exc()
                if child is not None and child.poll() is None and birth is not None:
                    process = psutil.Process(child.pid)
                    if process.create_time() == birth['birth']:
                        for member in process.children(recursive=True):
                            member.kill()
                        process.kill()
                    child.wait(timeout=10)
                raise
            finally:
                run.update(complete=True, ended=time.time(), seconds=time.monotonic()-started)
                if child is not None:
                    run['code'] = child.poll()
                save(path, state)
            assert absent(birth)
            if job['engine'] == 'ort':
                state['pairs'].append(dict(request=job['request'], incoming=job['incoming'], maximum=check_pair(spec, job['request'], job['incoming'])))
                save(path, state)
            print(json.dumps(dict(job=job['id'], completed=len(state['runs']), seconds=run['seconds'])), flush=True)
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); raise
    finally:
        state.update(complete=True, ended=time.time()); save(path, state); parent.cpu_affinity(previous_affinity)


if __name__ == '__main__':
    main()
