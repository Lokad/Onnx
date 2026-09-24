"""Bounded two-process observation of the pinned ORT application on the AMD VM."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import psutil

BASE = Path(__file__).resolve().parent
GIB = 1024**3


def pin(path):
    path = Path(path)
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf8')
    temp.replace(path)


def live(identity):
    try:
        p = psutil.Process(identity['pid'])
        return p.create_time() == identity['birth'] and p.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def size():
    return sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file())


def main():
    assert sys.platform == 'linux' and not sys.flags.optimize
    assert not (BASE/'state.json').exists()
    own = psutil.Process(); own.cpu_affinity([0])
    spec = read(BASE/'spec.json'); app = Path(spec['app'])
    assert psutil.boot_time() == spec['boot_time'] == 1789634288.0
    for name, wanted in spec['tools'].items():
        assert pin(BASE/name) == wanted
    assert pin(app/'payload.json') == spec['payload']
    assert pin(app/'collection.json') == spec['collection']
    collection = read(app/'collection.json')
    assert collection['terminal'] and collection['code'] == 0 and collection['input_error'] is None
    assert all(not live(i) for i in collection['identities'])
    sys.path.insert(0, str(app/'tools'))
    from protocol import verify
    from remote import idle  # This file runs as __main__; the imported module is the pinned app controller.
    idle()
    payload = verify(app)
    assert pin(sys.executable) == payload['interpreter']
    assert pin(app/'runtime/native.py') == spec['native_consumer']
    assert pin(app/'manifests/current-parakeet.json') == spec['manifest']
    import importlib.util
    module = importlib.util.spec_from_file_location('process_accounting', app/'runtime/campaign_processes.py')
    accounting = importlib.util.module_from_spec(module); module.loader.exec_module(accounting)
    environment = {k: v for k, v in os.environ.items() if not k.lower().startswith(('lokad_', 'dotnet_', 'complus_'))}
    environment.pop('PYTHONOPTIMIZE', None)
    environment.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1', PYTHONPATH=os.pathsep.join(payload['python_paths']))
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        environment[key] = '1'
    state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()),
                 started=time.time(), runs=[], spec=pin(BASE/'spec.json'))
    save(BASE/'state.json', state)
    try:
        capabilities = dict(perf=shutil.which('perf'), readelf=shutil.which('readelf'), nm=shutil.which('nm'))
        for name in ['perf_event_paranoid', 'kptr_restrict']:
            path = Path('/proc/sys/kernel')/name
            capabilities[name] = path.read_text().strip() if path.exists() else None
        save(BASE/'capabilities.json', capabilities)
        for mode in ['control', 'profile']:
            preflight = dict(available=psutil.virtual_memory().available, tmpfs=psutil.disk_usage(BASE).free)
            assert preflight['available'] >= 12*GIB and preflight['tmpfs'] >= 3*GIB
            before = accounting.snapshot()
            row = dict(mode=mode, complete=False, code=None, before=before, preflight=preflight, samples=0, peak_rss=0)
            state['runs'].append(row); save(BASE/'state.json', state)
            child = None; started = time.monotonic()
            try:
                with (BASE/(mode+'.stdout')).open('x') as out, (BASE/(mode+'.stderr')).open('x') as err, (BASE/(mode+'.resources.jsonl')).open('x') as log:
                    command = [sys.executable, '-B', str(BASE/'observer.py'), str(app/'runtime/native.py'),
                               str(app/'assets'), str(app/'manifests/current-parakeet.json'), str(BASE/mode), mode]
                    own.cpu_affinity([2])
                    try:
                        child = subprocess.Popen(command, cwd=BASE, env=environment, stdin=subprocess.DEVNULL,
                                                 stdout=out, stderr=err, start_new_session=True)
                    finally:
                        own.cpu_affinity([0])
                    p = psutil.Process(child.pid)
                    row['owner'] = dict(pid=p.pid, birth=p.create_time()); save(BASE/'state.json', state)
                    while child.poll() is None:
                        try:
                            assert p.create_time() == row['owner']['birth'] and not p.children(recursive=True)
                            affinities = []
                            for thread in p.threads():
                                try:
                                    affinities.append(sorted(os.sched_getaffinity(thread.id)))
                                except ProcessLookupError:
                                    pass
                            sample = dict(seconds=time.monotonic()-started, rss=p.memory_info().rss,
                                available=psutil.virtual_memory().available, tmpfs=psutil.disk_usage(BASE).free,
                                output=size(), affinities=affinities)
                        except psutil.NoSuchProcess:
                            continue
                        log.write(json.dumps(sample)+'\n'); log.flush()
                        row['samples'] += 1; row['peak_rss'] = max(row['peak_rss'], sample['rss'])
                        save(BASE/'state.json', state)
                        assert sample['seconds'] < 900 and sample['rss'] < 12*GIB
                        assert sample['available'] >= GIB and sample['tmpfs'] >= GIB and sample['output'] <= GIB
                        assert all(a == [2] for a in affinities)
                        time.sleep(.5)
                    row['code'] = child.wait(); assert row['code'] == 0, mode
                assert not live(row['owner'])
                row['after'] = accounting.snapshot()
                row['accounting'] = accounting.foreign_fraction(before, row['after'], own.pid)
                assert row['accounting']['valid'] and row['accounting']['foreign_cpu_fraction'] <= .01
            except BaseException:
                if 'owner' in row and live(row['owner']):
                    psutil.Process(row['owner']['pid']).kill()
                if child is not None:
                    child.wait(timeout=15)
                raise
            finally:
                row.update(complete=True, seconds=time.monotonic()-started,
                           code=None if child is None else child.poll())
                save(BASE/'state.json', state)
        verify(app)
        assert size() <= GIB
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); traceback.print_exc()
    finally:
        state.update(complete=True, ended=time.time()); save(BASE/'state.json', state)
    return state['code']


if __name__ == '__main__':
    raise SystemExit(main())
