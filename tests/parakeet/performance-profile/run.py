"""Run one full-corpus attribution worker with bounded local resources."""
import ast
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from prepare import ROOT, TOOLS, BASE, MANIFEST, SITE, pin, read, save, verify


def main():
    sys.path.insert(0, str(SITE)); import psutil
    prepared = read(BASE/'prepared.json'); assert prepared['passed']; verify(prepared['files'])
    assert not (BASE/'frozen.json').exists() and not (BASE/'state.json').exists()
    files = dict(prepared['files'])
    for path in TOOLS.glob('*.py'):
        ast.parse(path.read_text(encoding='utf8')); files[path.relative_to(ROOT).as_posix()] = pin(path)
    save(BASE/'frozen.json', dict(files=files, prepared=pin(BASE/'prepared.json')))
    own = psutil.Process(); previous = own.cpu_affinity(); own.cpu_affinity([0]); child = None
    state = dict(complete=False, code=None, phase='memory-preflight', supervisor=dict(pid=own.pid, birth=own.create_time()),
                 preflight_observations=[], samples=0, peak_rss=0)
    save(BASE/'state.json', state); start = time.monotonic()
    try:
        while True:
            value = dict(seconds=time.monotonic()-start, available=psutil.virtual_memory().available, disk=shutil.disk_usage(BASE).free)
            state['preflight_observations'].append(value); save(BASE/'state.json', state)
            assert value['seconds'] < 900 and value['disk'] >= 20*1024**3, 'Preflight refused before worker launch'
            if value['available'] >= 10*1024**3: break
            time.sleep(15)
        state['preflight'] = value; state['phase'] = 'profiling'; save(BASE/'state.json', state)
        env = {k: v for k, v in os.environ.items() if not k.lower().startswith(('lokad_', 'dotnet_', 'complus_'))}
        command = ['dotnet', str(BASE/'bin/Profile.dll'), str(ROOT), str(MANIFEST), str(BASE/'output')]
        with (BASE/'stdout.txt').open('x') as out, (BASE/'stderr.txt').open('x') as err, (BASE/'samples.jsonl').open('x') as samples:
            own.cpu_affinity([2])
            try: child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                                          creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW)
            finally: own.cpu_affinity([0])
            worker = psutil.Process(child.pid); state['worker'] = dict(pid=worker.pid, birth=worker.create_time()); save(BASE/'state.json', state)
            beginning = time.monotonic(); last_scan = -5.; artifact_bytes = 0
            while child.poll() is None:
                try:
                    assert worker.create_time() == state['worker']['birth'] and not worker.children(recursive=True)
                    seconds = time.monotonic()-beginning
                    if seconds-last_scan >= 5:
                        artifact_bytes = sum(p.stat().st_size for p in (BASE/'output').rglob('*') if p.is_file()); last_scan = seconds
                    value = dict(seconds=seconds, rss=worker.memory_info().rss, available=psutil.virtual_memory().available,
                                 disk=shutil.disk_usage(BASE).free, affinity=worker.cpu_affinity(), artifact_bytes=artifact_bytes)
                except psutil.NoSuchProcess:
                    if child.poll() is not None: break
                    raise
                samples.write(json.dumps(value)+'\n'); samples.flush(); state['samples'] += 1; state['peak_rss'] = max(state['peak_rss'], value['rss'])
                save(BASE/'state.json', state)
                assert value['seconds'] < 1800 and value['rss'] < 8*1024**3 and value['available'] >= 1024**3
                assert value['disk'] >= 20*1024**3 and value['affinity'] == [2] and value['artifact_bytes'] <= 1024**3
                time.sleep(.25)
            state['code'] = child.wait(); assert state['code'] == 0, state['code']
        verify(files); state.update(passed=True, phase='worker-complete')
    except BaseException:
        state.update(code=1, error=traceback.format_exc())
        if child is not None and child.poll() is None:
            worker = psutil.Process(child.pid)
            if worker.create_time() == state['worker']['birth']: worker.kill(); child.wait(timeout=15)
        raise
    finally:
        state['complete'] = True; save(BASE/'state.json', state); own.cpu_affinity(previous)
    print(json.dumps(state))


if __name__ == '__main__': main()
