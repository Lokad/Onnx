"""Run unchanged pyannote product attribution once, with bounded local resources."""
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
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/pyannote-performance-profile-20260921'
PRODUCT = ROOT / 'artifacts/whisper-memory-product-v2-20260921/source/src/Lokad.Onnx.CLI/bin/Release/net10.0'
MANIFEST = ROOT / 'artifacts/audio-ort-baseline-v2-20260919/inputs/pyannote.json'
sys.path.append(str(ROOT / 'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
import psutil


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf8')
    temporary.replace(path)


def main():
    BASE.mkdir(exist_ok=False)
    binary = BASE / 'bin'
    shutil.copytree(TOOLS / 'bin/Release/net10.0', binary)
    for path in PRODUCT.glob('*.dll'):
        shutil.copy2(path, binary / path.name)
    spec = json.loads(MANIFEST.read_text())
    files = {str(MANIFEST.relative_to(ROOT)): pin(MANIFEST)}
    for item in list(spec['models'].values()) + [c['pcm'] for c in spec['cases']]:
        actual = pin(ROOT / item['path'])
        assert actual == {k: item[k] for k in ('bytes', 'sha256')}
        files[item['path']] = actual
    for path in list(TOOLS.glob('*.cs')) + list(TOOLS.glob('*.csproj')) + [Path(__file__), ROOT / 'tests/Shared/NpySupport.cs'] + list(binary.iterdir()):
        if path.is_file():
            files[str(path.relative_to(ROOT))] = pin(path)
    assert pin(binary / 'Lokad.Onnx.dll')['sha256'] == 'd1f86a7346dcd70ebcc9ef7d9cd9633f05ad3a5275ca39f035c72325a0531fa4'
    save(BASE / 'frozen.json', dict(files=files,manifest=str(MANIFEST.relative_to(ROOT)),normal_runtime=True))
    own = psutil.Process(); previous = own.cpu_affinity(); own.cpu_affinity([0])
    state = dict(complete=False,code=None,supervisor=dict(pid=own.pid,birth=own.create_time()))
    child = None
    try:
        preflight = dict(available=psutil.virtual_memory().available,disk=shutil.disk_usage(BASE).free)
        state['preflight'] = preflight
        assert preflight['available'] >= 10*1024**3 and preflight['disk'] >= 20*1024**3, preflight
        command = ['dotnet',str(binary/'Profile.dll'),str(ROOT),str(MANIFEST),str(BASE/'output')]
        env = {k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
        with (BASE/'stdout.txt').open('x') as out, (BASE/'stderr.txt').open('x') as err, (BASE/'samples.jsonl').open('x') as log:
            own.cpu_affinity([2])
            try:
                child = subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err,
                    creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NO_WINDOW)
            finally:
                own.cpu_affinity([0])
            worker = psutil.Process(child.pid)
            state['worker'] = dict(pid=worker.pid,birth=worker.create_time())
            save(BASE/'state.json',state); started = time.monotonic()
            while child.poll() is None:
                try:
                    assert worker.create_time() == state['worker']['birth'] and not worker.children(recursive=True)
                    sample = dict(seconds=time.monotonic()-started,rss=worker.memory_info().rss,
                        available=psutil.virtual_memory().available,disk=shutil.disk_usage(BASE).free,affinity=worker.cpu_affinity())
                except psutil.NoSuchProcess:
                    continue
                log.write(json.dumps(sample)+'\n'); log.flush()
                assert sample['seconds'] < 900 and sample['rss'] < 8*1024**3 and sample['available'] >= 1024**3
                assert sample['disk'] >= 20*1024**3 and sample['affinity'] == [2]
                time.sleep(.25)
            state['code'] = child.wait()
            assert state['code'] == 0, state['code']
        for name, expected in files.items():
            assert pin(ROOT/name) == expected, name
        state['passed'] = True
    except BaseException:
        state.update(code=1,error=traceback.format_exc())
        if child is not None and child.poll() is None:
            worker = psutil.Process(child.pid)
            if worker.create_time() == state['worker']['birth']:
                worker.kill();child.wait(timeout=10)
        raise
    finally:
        state['complete'] = True
        save(BASE/'state.json',state)
        own.cpu_affinity(previous)
    print(json.dumps(state))


if __name__ == '__main__':
    main()
