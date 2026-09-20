"""Bounded sequential CPU0 workers, with original process identities retained."""
import argparse, subprocess, time
from common import *

def main():
    p = argparse.ArgumentParser(); p.add_argument('--artifact', required=True); a = p.parse_args()
    base = Path(a.artifact).resolve(); ps = psutil_module(); own = ps.Process(); own.cpu_affinity([0])
    spec = read(base / 'manifest.json'); verify(spec['files']); assert spec['limits'] == LIMITS
    state = dict(supervisor=dict(pid=own.pid, birth=own.create_time()), manifest=pin(base / 'manifest.json'), complete=False, code=None, runs=[])
    statepath = base / 'run.json'; write(statepath, state)
    def save(): statepath.write_text(json.dumps(state, indent=2), encoding='utf-8')
    child = None
    try:
        for engine in ['numpy', 'torch']:
            available = ps.virtual_memory().available; disk = ps.disk_usage(str(base)).free
            assert available >= LIMITS['preflight'] and disk >= LIMITS['disk']
            start = time.perf_counter(); run = dict(engine=engine, started=time.time(), preflight_available=available, preflight_disk=disk, complete=False, samples=0)
            with (base / (engine + '.stdout')).open('x') as stdout, (base / (engine + '.stderr')).open('x') as stderr:
                env = os.environ.copy(); env.update(THREADS)
                child = subprocess.Popen([sys.executable, '-X', 'utf8', '-B', str(Path(__file__).with_name('worker.py')), '--artifact', str(base), '--engine', engine],
                                         stdout=stdout, stderr=stderr, env=env, creationflags=subprocess.DETACHED_PROCESS)
                process = ps.Process(child.pid); run['worker'] = dict(pid=child.pid, birth=process.create_time()); state['runs'].append(run); save()
                with (base / (engine + '.samples.jsonl')).open('x', encoding='utf-8') as stream:
                    while child.poll() is None:
                        elapsed = time.perf_counter() - start
                        try:
                            assert process.create_time() == run['worker']['birth'] and process.cpu_affinity() == [0]
                            assert not process.children(recursive=True)
                            row = dict(seconds=elapsed, pid=process.pid, birth=process.create_time(), affinity=process.cpu_affinity(),
                                       rss=process.memory_info().rss, available=ps.virtual_memory().available)
                        except ps.NoSuchProcess:
                            assert child.wait(timeout=5) == 0
                            break
                        stream.write(json.dumps(row) + '\n'); stream.flush(); run['samples'] += 1
                        assert elapsed < LIMITS['seconds'] and row['rss'] < LIMITS['rss'] and row['available'] >= LIMITS['available']
                        time.sleep(.25)
                run.update(code=child.wait(), complete=True, ended=time.time(), seconds=time.perf_counter() - start)
                save(); assert run['code'] == 0
                print(json.dumps(dict(engine=engine, seconds=run['seconds'], samples=run['samples'])), flush=True)
        state['code'] = 0
    except BaseException as error:
        state['error'] = repr(error); state['code'] = 1
        if child is not None and child.poll() is None:
            process = ps.Process(child.pid)
            if process.create_time() == state['runs'][-1]['worker']['birth']:
                process.kill(); child.wait(timeout=10)
        raise
    finally:
        state['complete'] = True; save()

if __name__ == '__main__': main()
