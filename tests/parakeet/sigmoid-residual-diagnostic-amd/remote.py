"""Observe two unchanged application processes using retained capture/export tools."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

sys.path.insert(0, '/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil

BASE = ROOT = Path(__file__).resolve().parent
spec = json.loads((BASE/'spec.json').read_text())


def load(name, path):
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


common = load('retained_monitor', spec['common'])
read, save, pin, live, idle = common.read, common.save, common.pin, common.live, common.idle
DOTNET = common.DOTNET
LIMITS = dict(artifacts=spec['limits']['output'])


def terminal(identity):
    assert not live(identity), identity


def artifact_size():
    return sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file())


def verify():
    assert psutil.boot_time() == spec['boot']
    for name, wanted in spec['files'].items():
        assert pin(BASE/name) == wanted, name
    for name, wanted in spec['external'].items():
        assert pin(name) == wanted, name
    return spec


def thread_affinities(process):
    threads = []
    for item in process.threads():
        try:
            threads.append(dict(id=item.id, affinity=sorted(os.sched_getaffinity(item.id))))
        except ProcessLookupError:
            pass
    assert threads and all(t['affinity'] == process.cpu_affinity() for t in threads)
    return threads


def clean_env(role):
    env = {k: v for k, v in os.environ.items()
           if not k.lower().startswith(('lokad_', 'dotnet_', 'complus_', 'parakeet_phase_'))}
    env.pop('PYTHONOPTIMIZE', None)
    env.update(TMPDIR=str(BASE/'tmp'), PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1')
    env['PATH'] = str(Path(DOTNET).parent)+os.pathsep+env.get('PATH', '')
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        env[name] = '1'
    if role == 'target':
        env.update(spec['diagnostic_flags'])
        env.update(PARAKEET_PHASE_MODE='control',
                   PARAKEET_PHASE_CORE_SHA=spec['product']['Lokad.Onnx.dll']['sha256'],
                   PARAKEET_PHASE_DATA_SHA=spec['product']['Lokad.Onnx.Data.dll']['sha256'])
    return env


def preflight():
    assert psutil.virtual_memory().available >= spec['limits']['available_before']
    assert psutil.disk_usage(BASE).free >= spec['limits']['tmpfs_before']
    assert artifact_size() <= LIMITS['artifacts']


def export(state, name, command):
    """The same resource checks as capture, with one CPU0 export process."""
    preflight()
    row = dict(name=name, complete=False, code=None, processes={}, samples=0,
               peak_rss=0, command=list(map(str, command)))
    state['runs'].append(row)
    save(BASE/'state.json', state)
    child = None
    start = time.monotonic()
    try:
        with (BASE/'logs'/(name+'.log')).open('x') as output, (BASE/'logs'/(name+'.samples.jsonl')).open('x') as samples:
            child = subprocess.Popen(row['command'], cwd=BASE, env=clean_env('export'),
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            process = psutil.Process(child.pid)
            identity = dict(pid=child.pid, birth=process.create_time(), affinity=[0])
            row['processes']['export'] = identity
            save(BASE/'state.json', state)
            while child.poll() is None:
                members = []
                try:
                    assert process.create_time() == identity['birth'] and not process.children(recursive=True)
                    members.append(dict(pid=process.pid, birth=process.create_time(), role='export',
                        affinity=process.cpu_affinity(), threads=thread_affinities(process), rss=process.memory_info().rss))
                except psutil.NoSuchProcess:
                    pass
                sample = dict(seconds=time.monotonic()-start, members=members,
                    rss=sum(m['rss'] for m in members), available=psutil.virtual_memory().available,
                    disk=psutil.disk_usage(BASE).free, output_bytes=artifact_size())
                samples.write(json.dumps(sample)+'\n'); samples.flush()
                row['samples'] += 1
                row['peak_rss'] = max(row['peak_rss'], sample['rss'])
                save(BASE/'state.json', state)
                assert sample['seconds'] < 900 and sample['rss'] < spec['limits']['rss']
                assert min(sample['available'], sample['disk']) >= spec['limits']['minimum_free']
                assert sample['output_bytes'] <= LIMITS['artifacts']
                assert all(m['affinity'] == [0] for m in members)
                time.sleep(.25)
            row['code'] = child.wait()
            assert row['code'] == 0
            terminal(identity)
    except BaseException:
        row['error'] = traceback.format_exc()
        if child is not None:
            if child.poll() is None:
                assert live(row['processes']['export'])
                child.kill()
            child.wait(timeout=15)
        raise
    finally:
        row.update(complete=True, seconds=time.monotonic()-start)
        save(BASE/'state.json', state)


def main():
    assert sys.platform == 'linux' and not (BASE/'state.json').exists()
    own = psutil.Process(); own.cpu_affinity([0])
    idle(); verify(); preflight()
    (BASE/'logs').mkdir(); (BASE/'tmp').mkdir()
    state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()),
                 started=time.time(), boot=psutil.boot_time(), runs=[])
    save(BASE/'state.json', state)
    namespace = dict(globals())
    exec(compile((BASE/'pair.py.txt').read_text(), 'retained_pair_with_explicit_logging', 'exec'), namespace)
    pair = namespace['pair']
    app = Path(spec['app'])
    protocol = load('original_protocol', app/'runtime/protocol.py')
    accounting = load('original_accounting', app/'runtime/campaign_processes.py')
    manifest = read(app/'manifests/current-parakeet.json')
    try:
        for name in spec['jobs']:
            verify(); preflight()
            sampled = name == 'sampled'
            before = accounting.snapshot()
            pair(state, BASE/'state.json', name, [DOTNET, Path(spec['runtime'])/'SampledAudio.dll',
                app/'assets', app/'manifests/current-parakeet.json', BASE/name, 'timing', name],
                BASE/name, sampled, 11)
            row = state['runs'][-1]
            row.update(cpu_before=before, cpu_after=accounting.snapshot())
            row['accounting'] = accounting.foreign_fraction(row['cpu_before'], row['cpu_after'], own.pid)
            save(BASE/'state.json', state)
            assert row['accounting']['valid'] and row['accounting']['foreign_cpu_fraction'] <= .01
            value = read(BASE/name/'result.json')
            assert value['passed'] and value['flags'] == spec['diagnostic_flags']
            # Only these prospectively declared logging flags differ; raw evidence stays intact.
            protocol.validate_records(dict(value, flags={}), manifest, 'timing')
            assert value['sampled'] == sampled and value['processor_count'] == 1
            assert value['core_sha256'] == spec['product']['Lokad.Onnx.dll']['sha256']
            assert value['data_sha256'] == spec['product']['Lokad.Onnx.Data.dll']['sha256']
            assert value['runner_sha256'] == spec['runtime_files']['SampledAudio.dll']['sha256']
            print(name, '80 complete requests checked', flush=True)
        (BASE/'exports').mkdir()
        trace = BASE/'sampled/capture.nettrace'
        for format in ['Speedscope', 'Chromium']:
            export(state, format.lower(), [DOTNET, spec['tracer'], 'convert', trace,
                '--format', format, '--output', BASE/'exports'/format.lower()])
        export(state, 'events', [DOTNET, spec['exporter'], trace, BASE/'events'])
        verify()
        assert artifact_size() <= LIMITS['artifacts']
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); traceback.print_exc()
    finally:
        state.update(complete=True, ended=time.time()); save(BASE/'state.json', state)
    return state['code']


if __name__ == '__main__':
    raise SystemExit(main())
