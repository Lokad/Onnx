"""Preserve a metadata-rename failure and rerun only the public controls."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT/'tests/parakeet/performance-profile'
sys.path.insert(0, str(TOOLS))
import run
from prepare import BASE, SITE, pin, read, save, verify


def bounded_save(path, value):
    # Retry only a Windows metadata rename/sharing failure, never inference.
    for attempt in range(20):
        try:
            save(path, value)
            return
        except PermissionError:
            if attempt == 19: raise
            time.sleep(.05)


def main():
    sys.path.insert(0, str(SITE)); import psutil
    failed = read(BASE/'public-state.json')
    assert failed['complete'] and failed['code'] == 1 and failed['mode'] == 'public'
    assert 'PermissionError: [WinError 5]' in failed['error'] and 'temporary.replace(path)' in failed['error']
    for identity in (failed['supervisor'], failed['worker']):
        try: assert psutil.Process(identity['pid']).create_time() != identity['birth']
        except psutil.NoSuchProcess: pass
    verify(read(BASE/'frozen.json')['files'])
    trace = read(BASE/'trace-state.json'); assert trace['complete'] and trace['passed'] and trace['code'] == 0
    assert read(BASE/'trace-output/result.json')['passed']
    assert not (BASE/'public-output/result.json').exists()
    # All resolved move targets must stay in this one known artifact. Preserve
    # the exact failed bytes in their own attempt directory before new writes.
    destination = (BASE/'failed-public-attempt').resolve(); base = BASE.resolve()
    assert base.is_relative_to(ROOT.resolve()/'artifacts') and destination.is_relative_to(base)
    paths = [BASE/(name+suffix) for name, suffix in [('public-state', '.json'), ('public-stdout', '.txt'),
             ('public-stderr', '.txt'), ('public-samples', '.jsonl'), ('public-output', '')]]
    assert all(p.resolve().is_relative_to(base) and p.exists() for p in paths)
    assert not destination.exists(); destination.mkdir()
    original = {p.relative_to(base).as_posix(): pin(p) for root in paths for p in ([root] if root.is_file() else root.rglob('*')) if p.is_file()}
    for path in paths:
        target = destination/path.name; assert target.resolve().is_relative_to(destination) and not target.exists()
        shutil.move(str(path), str(target))
    for name, wanted in original.items(): assert pin(destination/name) == wanted
    shutil.copy2(Path(__file__), BASE/'public-resume.py')
    save(destination/'failure.json', dict(passed=False, reason='Atomic progress-file replacement was denied by Windows; no inference/numerical/resource failure',
        identities_verified_absent=[failed['supervisor'], failed['worker']], files=original,
        unchanged_trace=pin(BASE/'trace-output/result.json'), supervisor_scope='Only metadata PermissionError is retried for at most one second'))
    run.save = bounded_save
    sys.argv = [str(TOOLS/'run.py'), 'public']
    run.main()


if __name__ == '__main__': main()
