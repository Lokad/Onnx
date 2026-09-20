"""Build and run all new precision variants and two references, with fixed process guards."""
import argparse, subprocess, time
from shared import *

def main():
    p = argparse.ArgumentParser(); p.add_argument('--artifact', required=True); a = p.parse_args()
    base = Path(a.artifact).resolve(); spec = read(base / 'manifest.json'); verify(spec['files'])
    ps = psutil_module(); own = ps.Process(); own.cpu_affinity([0])
    state = dict(manifest=pin(base / 'manifest.json'), supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[], complete=False, code=None, limits=LIMITS)
    statepath = base / 'run.json'; write(statepath, state)
    def save(): statepath.write_text(json.dumps(state, indent=2), encoding='utf-8')
    env = {k: v for k, v in os.environ.items() if not k.lower().startswith(('lokad_', 'dotnet_', 'complus_'))}
    env.update(THREADS); env.update(DOTNET_PROCESSOR_COUNT='1', UseSharedCompilation='false', MSBUILDDISABLENODEREUSE='1',
                                   DOTNET_CLI_DO_NOT_USE_MSBUILD_SERVER='1', NUGET_PACKAGES=str(base / 'packages'))
    commands = [('build', ['dotnet', 'build', 'Precision.csproj', '-c', 'Release', '--tl:off', '--nologo', '-v', 'minimal', '--disable-build-servers', '-p:FrozenCorePath='+str(ROOT/spec['core'])])]
    dll = base / 'app/bin/Release/net10.0/Precision.dll'
    commands += [('managed', ['dotnet', str(dll), str(base)])]
    commands += [(engine, [sys.executable, '-X', 'utf8', '-B', str(Path(__file__).with_name('worker.py')), '--artifact', str(base), '--engine', engine]) for engine in ['numpy','torch']]
    try:
        for name, command in commands:
            available = ps.virtual_memory().available; disk = ps.disk_usage(str(base)).free
            assert available >= LIMITS['preflight'] and disk >= LIMITS['disk']
            run = dict(name=name, command=command, started=time.time(), members={}, complete=False, code=None, samples=0,
                       preflight_available=available, preflight_disk=disk)
            state['runs'].append(run); save(); start = time.perf_counter(); child = None
            try:
                with (base / (name + '.stdout')).open('x') as stdout, (base / (name + '.stderr')).open('x') as stderr, (base / (name + '.samples.jsonl')).open('x') as stream:
                    child = subprocess.Popen(command, cwd=base / 'app', env=env, stdout=stdout, stderr=stderr, creationflags=subprocess.DETACHED_PROCESS)
                    process = ps.Process(child.pid); run['child'] = dict(pid=child.pid, birth=process.create_time())
                    run['members'][str(child.pid)] = run['child']['birth']; save()
                    while child.poll() is None:
                        members = []
                        try:
                            assert process.create_time() == run['child']['birth']
                            for member in [process] + process.children(recursive=True):
                                try:
                                    birth = member.create_time(); key = str(member.pid)
                                    if key in run['members']: assert run['members'][key] == birth
                                    run['members'][key] = birth
                                    members.append(dict(pid=member.pid, birth=birth, rss=member.memory_info().rss, affinity=member.cpu_affinity()))
                                except ps.NoSuchProcess: pass
                        except ps.NoSuchProcess: pass
                        row = dict(seconds=time.perf_counter() - start, available=ps.virtual_memory().available, members=members)
                        stream.write(json.dumps(row) + '\n'); stream.flush(); run['samples'] += 1
                        assert row['seconds'] < LIMITS['seconds'] and sum(m['rss'] for m in members) < LIMITS['rss'] and row['available'] >= LIMITS['available']
                        assert all(m['affinity'] == [0] for m in members)
                        time.sleep(.25)
                    run['code'] = child.wait(); assert run['code'] == 0
                    deadline = time.perf_counter() + 10
                    while not all(absent(dict(pid=int(pid), birth=birth)) for pid, birth in run['members'].items()):
                        assert time.perf_counter() < deadline; time.sleep(.1)
            except BaseException as error:
                run['error'] = repr(error)
                for pid, birth in reversed(list(run['members'].items())):
                    try:
                        member = ps.Process(int(pid))
                        if member.create_time() == birth: member.kill()
                    except ps.NoSuchProcess: pass
                if child is not None: child.wait(timeout=10)
                raise
            finally:
                run.update(complete=True, ended=time.time(), seconds=time.perf_counter() - start); save()
            print(json.dumps(dict(stage=name, seconds=run['seconds'], samples=run['samples'])), flush=True)
        state['code'] = 0
    except BaseException as error: state.update(code=1, error=repr(error)); raise
    finally: state['complete'] = True; save()

if __name__ == '__main__': main()
