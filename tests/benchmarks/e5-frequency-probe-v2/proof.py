"""Bounded no-model proof of perf epoch and installed .NET clock compatibility."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import psutil
from counter import Counter, EVENTS, epoch, intervals

BASE = Path(__file__).resolve().parent


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def live(identity):
    try:
        p = psutil.Process(identity['pid'])
        return p.create_time() == identity['birth'] and p.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def main():
    own = psutil.Process(); own.cpu_affinity([0])
    assert not (BASE/'state.json').exists()
    state = dict(complete=False, code=None, owner=dict(pid=own.pid, birth=own.create_time()),
        started=time.time(), samples=[], identities={}, anchors=[], inference_calls=0,
        system_settings_changed=False)
    save(BASE/'state.json', state)
    children = {}; streams = []; counter = None; started = time.monotonic()

    def observe():
        assert time.monotonic()-started < 30
        members = []
        for label, child in children.items():
            if child.poll() is not None: continue
            try:
                parent = psutil.Process(child.pid)
                for p in [parent]+parent.children(recursive=True):
                    try:
                        if p.status() == psutil.STATUS_ZOMBIE: continue
                        birth = p.create_time()
                        assert state['identities'].get(str(p.pid), birth) == birth
                        state['identities'][str(p.pid)] = birth
                        affinity = p.cpu_affinity()
                        assert affinity == [2] if label == 'workload' else affinity in [[0], [2]]
                        members.append(dict(pid=p.pid, birth=birth, role=label, affinity=affinity,
                            rss=p.memory_info().rss))
                    except psutil.NoSuchProcess: pass
            except psutil.NoSuchProcess: pass
        size = sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file())
        sample = dict(seconds=time.monotonic()-started, members=members,
            rss=sum(m['rss'] for m in members), available=psutil.virtual_memory().available,
            tmpfs=psutil.disk_usage(BASE).free, output=size)
        assert sample['rss'] < 512*1024**2 and sample['output'] < 64*1024**2
        assert sample['available'] >= 1024**3 and sample['tmpfs'] >= 1024**3
        state['samples'].append(sample); save(BASE/'state.json', state)

    def spawn(label, command, cpu):
        handles = [(BASE/(label+suffix)).open('x') for suffix in ['.stdout', '.stderr']]
        streams.extend(handles)
        own.cpu_affinity([cpu])
        try:
            child = subprocess.Popen(command, stdout=handles[0], stderr=handles[1],
                stdin=subprocess.DEVNULL, start_new_session=True)
        finally: own.cpu_affinity([0])
        children[label] = child
        state['identities'][str(child.pid)] = psutil.Process(child.pid).create_time()
        state[label+'_command'] = command
        observe()
        return child

    try:
        assert psutil.boot_time() == 1789634288.0
        library = Path('/home/vermorel/.dotnet/shared/Microsoft.NETCore.App/10.0.8/libSystem.Native.so')
        native = ctypes.CDLL(str(library)).SystemNative_GetTimestamp
        native.argtypes = []; native.restype = ctypes.c_uint64
        brackets = []
        for i in range(1000):
            before = time.monotonic_ns(); actual = native(); after = time.monotonic_ns()
            assert before <= actual <= after
            brackets.append(dict(before_ns=before, native_ns=actual, after_ns=after))
        state['runtime_clock'] = dict(library=str(library), identity=pin(library),
            symbol='SystemNative_GetTimestamp', brackets=brackets)
        counter = Counter(BASE/'counters')
        before_launch = time.monotonic_ns()
        perf = spawn('frequency', counter.command(), 0)
        state['anchors'].append(counter.begin(before_launch)); observe()
        script = '''import json,time
from pathlib import Path
start=time.monotonic_ns();value=0
while time.monotonic_ns()-start<2300000000:
 for i in range(10000):value=(value+i)%104729
end=time.monotonic_ns()
Path('WORKLOAD_RESULT').write_text(json.dumps(dict(start_ns=start,end_ns=end,value=value)))
'''.replace('WORKLOAD_RESULT', str(BASE/'workload.json'))
        workload = spawn('workload', [sys.executable, '-B', '-c', script], 2)
        while workload.poll() is None:
            observe(); time.sleep(.025)
        assert workload.returncode == 0
        state['anchors'].append(counter.finish(perf)); observe()
        assert perf.returncode == 0
        state['epoch'] = epoch(state['anchors'])
        state['intervals'] = intervals(counter.output.read_text())
        work = json.loads((BASE/'workload.json').read_text()); state['workload'] = work
        full = []
        for before, after in zip(state['intervals'], state['intervals'][1:]):
            if (state['epoch']['lower_ns']+before['elapsed_ns'] >= work['start_ns'] and
                state['epoch']['upper_ns']+after['elapsed_ns'] <= work['end_ns']):
                full.append(after)
        assert full, 'No entire counter interval lies within the known workload'
        for group in full:
            for name in EVENTS[:3]:
                event = group['events'][name]
                assert event['count'] is not None and int(event['count']) > 0
                assert float(event['running_percent']) >= 99.9
        state['whole_workload_intervals'] = [row['elapsed_ns'] for row in full]
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc())
    finally:
        if counter is not None:
            if not (counter.folder/'stop').exists(): counter.stop()
        for label, child in children.items():
            if label == 'workload' and child.poll() is None: child.kill()
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Terminate only the exact recorded descendants of our perf process.
                identities = [dict(pid=int(pid), birth=birth) for pid,birth in state['identities'].items()]
                for identity in reversed(identities):
                    if live(identity):
                        subprocess.run(['sudo', '-n', '/bin/kill', '-KILL', str(identity['pid'])], check=True, timeout=5)
                child.wait(timeout=5)
        for stream in streams: stream.close()
        if counter is not None: counter.close()
        state['exitcodes'] = {label:child.returncode for label,child in children.items()}
        state['terminal'] = all(not live(dict(pid=int(pid),birth=birth)) for pid,birth in state['identities'].items())
        state.update(complete=True, ended=time.time()); save(BASE/'state.json', state)
    return state['code']


if __name__ == '__main__': raise SystemExit(main())
