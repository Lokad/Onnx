"""Explicit successor for the public control's insufficient RSS budget.

The original successful trace, consumer and frozen tools remain unchanged.
Only this new public worker uses a 12 GiB ceiling and 14 GiB preflight.
"""
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tests/parakeet/performance-profile'))
from prepare import BASE, SITE, pin, read, save as original_save, verify


def save(path, value):
    for attempt in range(20):
        try:
            original_save(path, value)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.05)


def terminal(identity):
    sys.path.insert(0, str(SITE))
    import psutil
    try:
        assert psutil.Process(identity['pid']).create_time() != identity['birth']
    except psutil.NoSuchProcess:
        pass


def prepare():
    verify(read(BASE/'frozen.json')['files'])
    trace = read(BASE/'trace-state.json')
    failed = read(BASE/'public-state.json')
    assert trace['complete'] and trace['passed'] and trace['code'] == 0
    assert read(BASE/'trace-output/result.json')['passed']
    assert failed['complete'] and failed['code'] == 1 and failed['mode'] == 'public'
    assert failed['peak_rss'] == 8723816448 > 8*1024**3
    samples = [json.loads(line) for line in (BASE/'public-samples.jsonl').read_text().splitlines()]
    assert len(samples) == failed['samples'] == 132
    assert samples[-1]['rss'] == failed['peak_rss']
    assert all(s['seconds'] < 1800 and s['available'] >= 1024**3 and s['disk'] >= 20*1024**3
               and s['affinity'] == [2] and s['artifact_bytes'] <= 1024**3 for s in samples)
    for state in (trace, failed):
        for key in ('supervisor', 'worker'):
            terminal(state[key])
    assert not (BASE/'public-output/result.json').exists()
    destination = (BASE/'failed-public-rss-attempt').resolve()
    base = BASE.resolve()
    assert base.is_relative_to(ROOT.resolve()/'artifacts') and destination.is_relative_to(base)
    paths = [BASE/(name+suffix) for name, suffix in [('public-state', '.json'), ('public-stdout', '.txt'),
             ('public-stderr', '.txt'), ('public-samples', '.jsonl'), ('public-output', '')]]
    assert all(p.resolve().is_relative_to(base) and p.exists() for p in paths)
    assert not destination.exists()
    destination.mkdir()
    original = {p.relative_to(base).as_posix(): pin(p) for root in paths
                for p in ([root] if root.is_file() else root.rglob('*')) if p.is_file()}
    for path in paths:
        target = destination/path.name
        assert target.resolve().is_relative_to(destination) and not target.exists()
        shutil.move(str(path), str(target))
    for name, wanted in original.items():
        assert pin(destination/name) == wanted
    save(destination/'failure.json', dict(passed=False, reason='Public worker exceeded its 8 GiB RSS ceiling',
        identities_verified_absent=[failed['supervisor'], failed['worker']], files=original,
        no_complete_public_result=True))
    files = {}
    for folder in (BASE/'trace-output', destination, BASE/'failed-public-attempt', Path(__file__).parent):
        for path in folder.rglob('*'):
            if path.is_file():
                files[path.relative_to(ROOT).as_posix()] = pin(path)
    for name in ('trace-state.json', 'trace-samples.jsonl', 'frozen.json', 'public-resume.py'):
        files[(BASE/name).relative_to(ROOT).as_posix()] = pin(BASE/name)
    save(BASE/'public-recovery-prepared.json', dict(passed=True, files=files,
        rss_limit_bytes=12*1024**3, preflight_available_bytes=14*1024**3,
        scope='Unchanged successful trace and consumer; new normal-runtime public control worker only',
        prior_public_failure=pin(destination/'failure.json')))
    print('Public successor prepared; original trace and both public failures preserved', flush=True)


def verify_recovery():
    prepared = read(BASE/'public-recovery-prepared.json')
    assert prepared['passed'] and prepared['rss_limit_bytes'] == 12*1024**3
    assert prepared['preflight_available_bytes'] == 14*1024**3
    verify(prepared['files'])
    for name in ('failed-public-attempt', 'failed-public-rss-attempt'):
        failed = read(BASE/name/'public-state.json')
        assert failed['complete'] and failed['code'] == 1
        for key in ('supervisor', 'worker'):
            terminal(failed[key])
