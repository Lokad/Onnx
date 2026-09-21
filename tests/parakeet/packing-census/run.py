"""Build and run a bounded prepared-mapping census of the profiled product."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'artifacts/parakeet-packing-census-20260921'
TRACE = ROOT/'artifacts/parakeet-performance-profile-v2-20260921'
PRODUCT = TRACE/'bin'
sys.path.insert(0, str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
import psutil


def read(path): return json.loads(path.read_text(encoding='utf8'))
def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def save(path, value):
    for attempt in range(20):
        try:
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf8')
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19: raise
            time.sleep(.05)


def main():
    assert pin(TRACE/'closed.json')['sha256'] == '6d8ce878f99acf291cb20348bba948bd1de7f128298c774cf94adf84e855700d'
    closed = read(TRACE/'closed.json'); assert closed['passed']
    files = {}
    for name, wanted in closed['files'].items():
        if name.startswith('models/') or name.startswith(str(PRODUCT.relative_to(ROOT)).replace('\\', '/')+'/'):
            assert pin(ROOT/name) == wanted
            files[name] = wanted
    assert len([name for name in files if name.startswith('models/')]) == 6
    BASE.mkdir(); source = BASE/'source'; source.mkdir()
    for name in ('Program.cs', 'Census.csproj'): shutil.copy2(Path(__file__).parent/name, source/name)
    env = {k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_', 'dotnet_', 'complus_'))}
    flags = ['--tl:off', '--nologo', '-v', 'minimal', '-p:EnableSourceControlManagerQueries=false', '-p:EnableSourceLink=false', '-p:UseSharedCompilation=false', '-nr:false']
    commands = [['dotnet', 'restore', 'Census.csproj', *flags, '--source', str(ROOT/'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'), '--packages', str(BASE/'packages'), '-p:NuGetAudit=false'],
        ['dotnet', 'build', 'Census.csproj', '-c', 'Release', *flags, '--no-restore', '--disable-build-servers', '-p:FrozenProductDirectory='+str(PRODUCT)]]
    builds = []
    for label, command in zip(('restore', 'build'), commands, strict=True):
        with (BASE/(label+'.log')).open('x') as log:
            code = subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300).returncode
        builds.append(dict(label=label, command=command, code=code)); save(BASE/'builds.json', builds)
        assert code == 0, label
    shutil.copytree(source/'bin/Release/net10.0', BASE/'bin')
    for path in PRODUCT.glob('*.dll'): shutil.copy2(path, BASE/'bin'/path.name)
    for folder in (Path(__file__).parent, source, BASE/'bin'):
        for path in folder.iterdir():
            if path.is_file(): files[path.relative_to(ROOT).as_posix()] = pin(path)
    files[(TRACE/'closed.json').relative_to(ROOT).as_posix()] = pin(TRACE/'closed.json')
    save(BASE/'prepared.json', dict(passed=True, files=files))
    own = psutil.Process(); previous = own.cpu_affinity(); own.cpu_affinity([0]); child = None
    state = dict(complete=False, code=None, supervisor=dict(pid=own.pid,birth=own.create_time()), samples=0, peak_rss=0)
    start = time.monotonic()
    try:
        while psutil.virtual_memory().available < 10*1024**3:
            assert time.monotonic()-start < 900, 'Preflight memory refusal'
            time.sleep(15)
        state['preflight_available'] = psutil.virtual_memory().available
        assert shutil.disk_usage(BASE).free > 20*1024**3
        with (BASE/'stdout.txt').open('x') as out, (BASE/'stderr.txt').open('x') as err, (BASE/'samples.jsonl').open('x') as samples:
            own.cpu_affinity([2])
            try:
                child = subprocess.Popen(['dotnet', str(BASE/'bin/Census.dll'), str(ROOT/'models/parakeet-tdt-0.6b-v3'), str(BASE/'result.json')],
                    cwd=ROOT, env=env, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                    creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW)
            finally: own.cpu_affinity([0])
            worker = psutil.Process(child.pid); state['worker'] = dict(pid=worker.pid,birth=worker.create_time()); start = time.monotonic()
            while child.poll() is None:
                try:
                    assert worker.create_time() == state['worker']['birth'] and not worker.children(recursive=True)
                    row = dict(seconds=time.monotonic()-start,rss=worker.memory_info().rss,available=psutil.virtual_memory().available,
                        affinity=worker.cpu_affinity(),disk=shutil.disk_usage(BASE).free)
                except psutil.NoSuchProcess:
                    if child.poll() is not None: break
                    raise
                samples.write(json.dumps(row)+'\n'); samples.flush(); state['samples'] += 1; state['peak_rss'] = max(state['peak_rss'],row['rss'])
                save(BASE/'state.json',state)
                assert row['seconds'] < 180 and row['rss'] < 8*1024**3 and row['available'] >= 1024**3 and row['affinity'] == [2] and row['disk'] > 20*1024**3
                time.sleep(.25)
            state['code'] = child.wait(); assert state['code'] == 0
        for name,wanted in files.items(): assert pin(ROOT/name) == wanted,name
        assert read(BASE/'result.json')['passed']
        state['passed'] = True
    except BaseException:
        state.update(code=1,error=traceback.format_exc())
        if child is not None and child.poll() is None:
            worker = psutil.Process(child.pid)
            if worker.create_time() == state['worker']['birth']: worker.kill();child.wait(timeout=15)
        raise
    finally:
        state['complete'] = True; save(BASE/'state.json',state); own.cpu_affinity(previous)
    print(json.dumps(state))


if __name__ == '__main__': main()
