"""Run one full-corpus attribution worker with bounded local resources."""
import ast
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from protocol import ROOT, BASE, SITE, pin, read, save, verify, prepare, verify_recovery
from prepare import TOOLS, MANIFEST


def main():
    sys.path.insert(0, str(SITE)); import psutil
    assert len(sys.argv) == 2 and sys.argv[1] == 'public'; mode = 'public'
    prepare(); verify_recovery()
    state_path = BASE/(mode+'-state.json'); output = BASE/(mode+'-output')
    prepared = read(BASE/'prepared.json'); assert prepared['passed']; verify(prepared['files'])
    assert not state_path.exists()
    files = dict(prepared['files'])
    for path in TOOLS.glob('*.py'):
        ast.parse(path.read_text(encoding='utf8')); files[path.relative_to(ROOT).as_posix()] = pin(path)
    if mode == 'trace':
        assert not (BASE/'frozen.json').exists(); save(BASE/'frozen.json', dict(files=files, prepared=pin(BASE/'prepared.json')))
    else:
        frozen = read(BASE/'frozen.json'); assert frozen['files'] == files and frozen['prepared'] == pin(BASE/'prepared.json')
        prior = read(BASE/'trace-state.json'); assert prior['complete'] and prior['passed'] and prior['code'] == 0
        for identity in (prior['supervisor'], prior['worker']):
            try: assert psutil.Process(identity['pid']).create_time() != identity['birth']
            except psutil.NoSuchProcess: pass
    own = psutil.Process(); previous = own.cpu_affinity(); own.cpu_affinity([0]); child = None
    state = dict(complete=False, code=None, mode=mode, phase='memory-preflight', supervisor=dict(pid=own.pid, birth=own.create_time()),
                 preflight_observations=[], samples=0, peak_rss=0)
    save(state_path, state); start = time.monotonic()
    try:
        while True:
            value = dict(seconds=time.monotonic()-start, available=psutil.virtual_memory().available, disk=shutil.disk_usage(BASE).free)
            state['preflight_observations'].append(value); save(state_path, state)
            assert value['seconds'] < 900 and value['disk'] >= 20*1024**3, 'Preflight refused before worker launch'
            if value['available'] >= 14*1024**3: break
            time.sleep(15)
        state['preflight'] = value; state['phase'] = 'profiling'; save(state_path, state)
        env = {k: v for k, v in os.environ.items() if not k.lower().startswith(('lokad_', 'dotnet_', 'complus_'))}
        command = ['dotnet', str(BASE/'bin/Profile.dll'), str(ROOT), str(MANIFEST), str(output), mode]
        with (BASE/(mode+'-stdout.txt')).open('x') as out, (BASE/(mode+'-stderr.txt')).open('x') as err, (BASE/(mode+'-samples.jsonl')).open('x') as samples:
            own.cpu_affinity([2])
            try: child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                                          creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW)
            finally: own.cpu_affinity([0])
            worker = psutil.Process(child.pid); state['worker'] = dict(pid=worker.pid, birth=worker.create_time()); save(state_path, state)
            beginning = time.monotonic(); last_scan = -5.; artifact_bytes = 0
            while child.poll() is None:
                try:
                    assert worker.create_time() == state['worker']['birth'] and not worker.children(recursive=True)
                    seconds = time.monotonic()-beginning
                    if seconds-last_scan >= 5:
                        artifact_bytes = sum(p.stat().st_size for p in output.rglob('*') if p.is_file()); last_scan = seconds
                    value = dict(seconds=seconds, rss=worker.memory_info().rss, available=psutil.virtual_memory().available,
                                 disk=shutil.disk_usage(BASE).free, affinity=worker.cpu_affinity(), artifact_bytes=artifact_bytes)
                except psutil.NoSuchProcess:
                    if child.poll() is not None: break
                    raise
                samples.write(json.dumps(value)+'\n'); samples.flush(); state['samples'] += 1; state['peak_rss'] = max(state['peak_rss'], value['rss'])
                save(state_path, state)
                assert value['seconds'] < 1800 and value['rss'] < 12*1024**3 and value['available'] >= 1024**3
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
        state['complete'] = True; save(state_path, state); own.cpu_affinity(previous)
    print(json.dumps(state))


if __name__ == '__main__': main()
