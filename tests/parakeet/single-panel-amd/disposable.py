"""Remove only unpinned build caches before inference."""
import shutil
from candidate_protocol import pin, read, write


def build_caches(base):
    base = base.resolve(); spec = read(base / 'payload.json'); built = read(base / 'campaign/built-files.json')
    protected = {(base / name).resolve() for name in [*spec['files'], *built]}
    targets = [base / 'work' / name for name in ['packages', 'cli-home', 'cache', 'nuget-http-cache', 'tmp']]
    targets += list((base / 'source').rglob('obj')) + list((base / 'il-bridge').rglob('obj'))
    entries = {}
    for target in targets:
        assert target.is_relative_to(base) and target.resolve().is_relative_to(base) and target != base
        assert not target.is_symlink()
        if not target.exists(): continue
        assert target.is_dir() and not any(p == target or p.is_relative_to(target) for p in protected)
        for path in target.rglob('*'):
            assert not path.is_symlink() and path.resolve().is_relative_to(target)
            if path.is_file(): entries[path.relative_to(base).as_posix()] = pin(path)
    # Original source/consumer files and every built binary remain present.
    write(base / 'campaign/build-cache-cleanup.json', dict(passed=True, files=entries,
        targets=[p.relative_to(base).as_posix() for p in targets], bytes=sum(v['bytes'] for v in entries.values())))
    for target in targets:
        if target.exists(): shutil.rmtree(target)
    for name in ['tmp', 'cache']: (base / 'work' / name).mkdir()
