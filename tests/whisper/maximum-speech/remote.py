"""Finite AMD maximum-speech replay, reusing verified closed payloads."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import subprocess
import time
import traceback

path = Path(__file__).with_name('process_support.py')
if not path.exists():
    path = Path(__file__).resolve().parents[2]/'parakeet/recording-amd/remote.py'
spec = importlib.util.spec_from_file_location('retained_process_support',path)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
h.RSS_LIMIT, h.TIME_LIMIT = 27*1024**3//2, 3600


def install(base):
    bundle = h.read(base/'bundle.json')
    assert not (base/'installed.json').exists()
    allowed = [Path('/home/vermorel/Onnx/artifacts')/name for name in (
        'parakeet-recording-amd-20260919','whisper-recording-amd-v2-20260919')]
    for name,source in bundle['borrowed'].items():
        original = Path(source)
        assert not original.is_symlink() and any(original.resolve().is_relative_to(root) for root in allowed)
        destination = h.safe_path(base,name)
        assert not destination.exists()
        h.verify_file(original,bundle['files'][name])
        destination.parent.mkdir(parents=True,exist_ok=True)
        os.link(original,destination)
    h.verify(base)
    for name,pin in h.read(base/'reference/assets.json')['files'].items():
        h.verify_file(Path(bundle['models'])/name,pin)
    v = os.statvfs(base)
    assert v.f_bavail*v.f_frsize >= 8*1024**2
    h.write_new(base/'installed.json',dict(bundle_sha256=h.sha(base/'bundle.json'),installed_at=time.time(),
        available_bytes=v.f_bavail*v.f_frsize,borrowed=bundle['borrowed']))
    print('All frozen assets, input, binaries and payload verified.',flush=True)


def run(base):
    h.verify(base)
    assert h.read(base/'installed.json')['bundle_sha256'] == h.sha(base/'bundle.json')
    out = base/'result'
    out.mkdir()
    os.sched_setaffinity(0,{0})
    identity = dict(schema=1,supervisor=h.proc(os.getpid()),started=time.time(),complete=False,runs=[],
        bundle_sha256=h.sha(base/'bundle.json'),limits=dict(rss=h.RSS_LIMIT,seconds=h.TIME_LIMIT,available_memory=h.AVAILABLE_MIN))
    def save():
        temp = out/'identity.tmp'
        temp.write_text(json.dumps(identity,indent=2)+'\n',encoding='utf-8')
        temp.replace(out/'identity.json')
    code = 2
    try:
        save()
        (out/'cpuinfo.txt').write_text(Path('/proc/cpuinfo').read_text())
        (out/'dotnet-info.txt').write_text(subprocess.check_output(['dotnet','--info'],text=True))
        command = ['dotnet',str(base/'bin/RecordingReplay.dll'),h.read(base/'bundle.json')['models'],
            str(base/'inputs/inputs.json'),str(base/'short/manifest.json'),str(out/'managed')]
        clean = {k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
        row = dict(name='managed',command=command,started=time.time(),code=None,samples=0,peak_rss=0,members={})
        start = time.monotonic()
        with (out/'managed.stdout').open('x') as stdout, (out/'managed.stderr').open('x') as stderr, (out/'managed-samples.jsonl').open('x') as samples:
            os.sched_setaffinity(0,{2})
            try:
                child = subprocess.Popen(command,cwd=base,env=clean,stdout=stdout,stderr=stderr,start_new_session=True)
            finally:
                os.sched_setaffinity(0,{0})
            first = h.proc(child.pid)
            assert first
            row.update(pid=child.pid,start=first['start'])
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
                    h.check_sample(sample,child.pid,first['start'])
                    time.sleep(.5)
            finally:
                row['code'] = h.stop(child,first['start'])
                row.update(seconds=time.monotonic()-start,ended=time.time())
                save()
        assert row['code'] == 0 and row['seconds'] < h.TIME_LIMIT
        h.verify(base)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('install','launch','run','collect'))
    parser.add_argument('base',type=Path)
    args = parser.parse_args()
    return dict(install=install,launch=h.launch,run=run,collect=h.collect)[args.action](args.base.resolve()) or 0


if __name__ == '__main__':
    raise SystemExit(main())
