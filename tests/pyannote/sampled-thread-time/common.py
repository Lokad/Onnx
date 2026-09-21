"""Bounded owned-process capture, with normal model startup and PID attachment."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/pyannote-sampled-thread-time-20260921'
QUALIFIED = ROOT / 'artifacts/pyannote-convolution-pool-applications-v2-20260921'
INPUT = ROOT / 'artifacts/audio-ort-baseline-v2-20260919/inputs/pyannote.json'
FEED = ROOT / 'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
TRACE_SOURCE = Path('C:/Users/JoannesVermorel/.dotnet/tools/.store/dotnet-trace/10.0.745401/dotnet-trace/10.0.745401/tools/net8.0/any')
MONITOR = ROOT / 'tests/parakeet/packing-budgets/common.py'
spec = importlib.util.spec_from_file_location('sampled_thread_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)
monitor.BASE = BASE
pin, read, save, verify, terminal, psutil = monitor.pin, monitor.read, monitor.save, monitor.verify, monitor.terminal, monitor.psutil
CORE = '0d098ba5fd3fd8799bb1dd018148123f296802c769db5f6148a1a5fa9d80118e'
DATA = '1d34666456a5da749dc3b40ee25621af0bed806c9167f3ad12e96d0736dca662'


def rel(path):
    return path.relative_to(ROOT).as_posix()


def clean_env():
    return {k: v for k, v in os.environ.items() if not k.lower().startswith(('lokad_', 'dotnet_', 'complus_'))}


def verify_spec(specification):
    verify(specification['files'])
    for name, expected in specification.get('external_files', {}).items():
        assert pin(Path(name)) == expected, name


def new_state():
    own = psutil.Process()
    return dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])


def pair(state, state_path, name, command, output, sampled, preflight_gib):
    assert not output.exists()
    own = psutil.Process()
    previous_affinity = own.cpu_affinity()
    own.cpu_affinity([0])
    row = dict(name=name, sampled=sampled, complete=False, code=None, target_command=list(map(str, command)),
        members={}, processes={}, samples=0, peak_rss=0, preflight_observations=[])
    state['runs'].append(row)
    save(state_path, state)
    processes, handles = {}, []
    start = time.monotonic()
    last_scan, output_bytes = -5., 0

    def spawn(role, arguments, affinity):
        log = (BASE / 'logs' / (name + '-' + role + '.log')).open('x', encoding='utf8')
        handles.append(log)
        own.cpu_affinity(affinity)
        try:
            child = subprocess.Popen(list(map(str, arguments)), cwd=ROOT, env=clean_env(),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW)
        finally:
            own.cpu_affinity([0])
        p = psutil.Process(child.pid)
        identity = dict(pid=p.pid, birth=p.create_time(), affinity=affinity)
        row['processes'][role] = identity
        row['members'][str(p.pid)] = dict(role=role, birth=p.create_time(), affinity=affinity)
        processes[role] = child
        save(state_path, state)
        return child

    try:
        while True:
            observation = dict(seconds=time.monotonic()-start, available=psutil.virtual_memory().available, disk=shutil.disk_usage(BASE).free)
            row['preflight_observations'].append(observation)
            save(state_path, state)
            assert observation['seconds'] < 900 and observation['disk'] >= 20*1024**3
            if observation['available'] >= preflight_gib*1024**3:
                break
            time.sleep(15)
        row['preflight'] = observation
        spawn('target', command, [2])
        start = time.monotonic()
        with (BASE / 'logs' / (name + '.samples.jsonl')).open('x', encoding='utf8') as samples:
            while True:
                if (output / 'ready.json').exists() and 'ready' not in row:
                    ready = read(output / 'ready.json')
                    identity = row['processes']['target']
                    assert ready['pid'] == identity['pid'] and abs(ready['birth_milliseconds']/1000 - identity['birth']) < .002
                    assert ready['affinity'] == 4 and ready['runtime'] == '10.0.12' and not ready['flags']
                    row['ready'] = ready
                    if sampled:
                        trace_command = ['dotnet', BASE / 'tracer/dotnet-trace.dll', 'collect',
                            '--process-id', str(identity['pid']), '--profile', 'dotnet-common,dotnet-sampled-thread-time',
                            '--providers', 'Lokad-Pyannote-Diagnostic:0xffffffffffffffff:4', '--buffersize', '64',
                            '--duration', '00:00:15:00', '--output', output / 'capture.nettrace']
                        row['collector_command'] = list(map(str, trace_command))
                        spawn('collector', trace_command, [0])
                    else:
                        row['released_at'] = time.perf_counter_ns()
                        save(output / 'release.json', dict(pid=identity['pid'], sampled=False))
                if sampled and 'released_at' not in row and (output / 'collector-enabled.json').exists():
                    assert 'collector' in processes and processes['collector'].poll() is None
                    enabled = read(output / 'collector-enabled.json')
                    assert enabled['pid'] == row['processes']['target']['pid'] and enabled['enabled']
                    row['enabled'] = enabled
                    row['released_at'] = time.perf_counter_ns()
                    save(output / 'release.json', dict(pid=enabled['pid'], sampled=True))
                members = []
                for role, child in processes.items():
                    if child.poll() is not None:
                        continue
                    try:
                        p = psutil.Process(child.pid)
                        identity = row['processes'][role]
                        assert p.create_time() == identity['birth'] and not p.children(recursive=True)
                        cpu = p.cpu_times()
                        members.append(dict(pid=p.pid, birth=p.create_time(), role=role,
                            affinity=p.cpu_affinity(), rss=p.memory_info().rss, cpu_user=cpu.user, cpu_system=cpu.system))
                    except psutil.NoSuchProcess:
                        assert child.poll() is not None
                elapsed = time.monotonic()-start
                if elapsed-last_scan >= 2:
                    output_bytes = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
                    last_scan = elapsed
                sample = dict(seconds=elapsed, rss=sum(m['rss'] for m in members), available=psutil.virtual_memory().available,
                    disk=shutil.disk_usage(BASE).free, output_bytes=output_bytes, members=members)
                samples.write(json.dumps(sample)+'\n')
                samples.flush()
                row['samples'] += 1
                row['peak_rss'] = max(row['peak_rss'], sample['rss'])
                save(state_path, state)
                assert elapsed < 900 and sample['rss'] < 8*1024**3 and sample['available'] >= 1024**3 and sample['disk'] >= 20*1024**3
                assert output_bytes <= 1024**3 and all(m['affinity'] == ([2] if m['role']=='target' else [0]) for m in members)
                assert elapsed < 180 or 'released_at' in row, 'Collector barrier deadline'
                if all(child.poll() is not None for child in processes.values()):
                    break
                for role, child in processes.items():
                    if child.poll() is not None:
                        assert child.returncode == 0, (role, child.returncode)
                time.sleep(.25)
        row['exit_codes'] = {role: child.wait() for role, child in processes.items()}
        assert row['exit_codes'] == ({'target': 0, 'collector': 0} if sampled else {'target': 0}) and 'released_at' in row
        assert read(output / 'result.json')['passed']
        for identity in row['processes'].values():
            terminal(identity)
        row['code'] = 0
        return row
    except BaseException:
        row.update(code=1, error=traceback.format_exc())
        for identity in reversed(list(row['processes'].values())):
            try:
                p = psutil.Process(identity['pid'])
                if p.create_time() == identity['birth']:
                    p.kill()
            except psutil.NoSuchProcess:
                pass
        for child in processes.values():
            child.wait(timeout=15)
        raise
    finally:
        row.update(complete=True, seconds=time.monotonic()-start)
        save(state_path, state)
        for handle in handles:
            handle.close()
        own.cpu_affinity(previous_affinity)
