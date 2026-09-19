"""Finite Whisper Linux recording replay using preserved process-accounting helpers."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import subprocess
import time
import traceback


def load_support():
    path = Path(__file__).with_name('process_support.py')
    if not path.exists():
        path = Path(__file__).resolve().parents[2]/'parakeet/recording-amd/remote.py'
    spec = importlib.util.spec_from_file_location('recording_process_support', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


h = load_support()
h.RSS_LIMIT = 27 * 1024**3 // 2
BORROW_ROOT = Path('/home/vermorel/Onnx/artifacts/parakeet-recording-amd-20260919')
BORROW_BUNDLE = '04ea92c5a625cb6e2128be6fca20aa388ed43f533879ce3dfb5b2415327ee068'
BORROW_COLLECTION = 'eb5ea54c2a9812baa87cc89da2ce65a2cf3ede485965a345d4a7af106d70805a'


def install(base):
    bundle = h.read(base/'bundle.json')
    assert not (base/'installed.json').exists()
    assert h.sha(BORROW_ROOT/'bundle.json') == BORROW_BUNDLE
    assert h.sha(BORROW_ROOT/'collection.json') == BORROW_COLLECTION
    prior = h.read(BORROW_ROOT/'collection.json')
    assert prior['complete'] and prior['code'] == 0
    for item in prior['terminal_processes']:
        live = h.proc(item['pid'])
        assert live is None or live['start'] != item['start'], 'Prior recording process remains'
    borrowed = {}
    for name, source in bundle['borrowed'].items():
        destination = h.safe_path(base, name)
        original = h.safe_path(BORROW_ROOT/'inputs', source)
        assert original.parent == (BORROW_ROOT/'inputs').resolve() and not original.is_symlink()
        assert not destination.exists()
        h.verify_file(original, bundle['files'][name])
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(original, destination)
        borrowed[name] = dict(source=str(original), sha256=h.sha(original))
    h.verify(base)
    assets = h.read(base/'reference/assets.json')
    models = Path(bundle['models'])
    for name, pin in assets['files'].items():
        h.verify_file(models/name, pin)
    v = os.statvfs(base)
    h.write_new(base/'installed.json', dict(installed_at=time.time(), bundle_sha256=h.sha(base/'bundle.json'),
        borrowed=borrowed, model_files=len(assets['files']), available_bytes=v.f_bavail*v.f_frsize))
    print('All logical payload files, borrowed inputs and fourteen model assets verified.', flush=True)


def run(base):
    h.verify(base)
    assert h.read(base/'installed.json')['bundle_sha256'] == h.sha(base/'bundle.json')
    out = base/'result'
    out.mkdir()
    os.sched_setaffinity(0, {0})
    identity = dict(schema=1, supervisor=h.proc(os.getpid()), started=time.time(), complete=False, runs=[],
        bundle_sha256=h.sha(base/'bundle.json'), limits=dict(rss=h.RSS_LIMIT, seconds=1800, available_memory=256*1024**2))
    def save():
        path = out/'identity.tmp'
        path.write_text(json.dumps(identity, indent=2)+'\n', encoding='utf-8')
        path.replace(out/'identity.json')
    models = h.read(base/'bundle.json')['models']
    jobs = [
        ('managed', ['dotnet', str(base/'bin/RecordingReplay.dll'), models, str(base/'inputs/inputs.json'),
                     str(base/'short/manifest.json'), str(out/'managed')]),
        ('cli-connected', ['dotnet', str(base/'bin/Lokad.Onnx.CLI.dll'), 'transcribe', models,
                           str(base/'inputs/connected.wav'), '--language=en', '--recording', '--json']),
    ]
    clean = {k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_', 'dotnet_', 'complus_'))}
    code = 2
    try:
        save()
        (out/'cpuinfo.txt').write_text(Path('/proc/cpuinfo').read_text())
        (out/'dotnet-info.txt').write_text(subprocess.check_output(['dotnet','--info'],text=True))
        for name, command in jobs:
            h.verify(base)
            row = dict(name=name, command=command, started=time.time(), code=None, samples=0, peak_rss=0, members={})
            start = time.monotonic()
            with (out/(name+'.stdout')).open('x') as stdout, (out/(name+'.stderr')).open('x') as stderr, (out/(name+'-samples.jsonl')).open('x') as samples:
                os.sched_setaffinity(0,{2})
                try:
                    child = subprocess.Popen(command,cwd=base,env=clean,stdout=stdout,stderr=stderr,start_new_session=True)
                finally:
                    os.sched_setaffinity(0,{0})
                first = h.proc(child.pid)
                assert first, 'Missing initial worker identity'
                birth = first['start']
                row.update(pid=child.pid,start=birth)
                identity['runs'].append(row)
                save()
                try:
                    while child.poll() is None:
                        sample = dict(seconds=time.monotonic()-start,members=h.members(child.pid),available_memory=h.available())
                        samples.write(json.dumps(sample)+'\n')
                        samples.flush()
                        row['samples'] += 1
                        row['peak_rss'] = max(row['peak_rss'],sum(m['rss'] for m in sample['members']))
                        for member in sample['members']:
                            row['members'][str(member['pid'])] = member['start']
                        save()
                        h.check_sample(sample,child.pid,birth)
                        time.sleep(.5)
                finally:
                    row['code'] = h.stop(child,birth)
                    row.update(seconds=time.monotonic()-start,ended=time.time())
                    save()
            assert row['code'] == 0 and row['seconds'] < 1800, name
            h.verify(base)
            print(name,'complete',row['seconds'],row['peak_rss'],flush=True)
        identity['complete'] = True
        code = 0
    except BaseException:
        identity['error'] = traceback.format_exc()
        traceback.print_exc()
    finally:
        identity['ended'] = time.time()
        save()
        (base/'complete.txt').write_text(str(code)+'\n')
    return code


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action',choices=('install','launch','run','collect'))
    p.add_argument('base',type=Path)
    a = p.parse_args()
    method = dict(install=install,launch=h.launch,run=run,collect=h.collect)[a.action]
    return method(a.base.resolve()) or 0


if __name__ == '__main__':
    raise SystemExit(main())
