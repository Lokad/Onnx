"""Additional storage checks; the original request/resource checks stay intact."""
from pathlib import Path, PurePosixPath
import os

STORAGE = dict(preflight=3*1024**3, remaining=512*1024**2, parent='/dev/shm', filesystem='tmpfs')
ENGINES = ['managed', 'ort', 'ort', 'managed']
PENDING = "**Whisper's matched AMD comparison is incomplete:**"
NEXT = '### Audio: Windows Microsoft ONNX Runtime baselines'


def safe_member(name):
    assert isinstance(name, str) and name and '\\' not in name
    value = PurePosixPath(name)
    assert not value.is_absolute() and '..' not in value.parts and '.' not in name.split('/')
    assert str(value) == name and ':' not in name and not name.endswith('/')
    return name


def validate_storage(value, preflight=False):
    assert value['filesystem'] == STORAGE['filesystem']
    assert value['mount'] == STORAGE['parent'] and value['device'] == value['mount_device'] != value['root_device']
    path = PurePosixPath(value['base'])
    assert path.parent == PurePosixPath(STORAGE['parent']) and str(path) == value['base']
    assert path.name.startswith('lokad-whisper-storage-') and '..' not in path.parts
    assert type(value['free']) is int and value['free'] >= STORAGE['preflight' if preflight else 'remaining']
    assert type(value['root_free']) is int and value['root_free'] >= 0


def observe_storage(base):
    """Bind the actual write destination to the /dev/shm mount, not a path label."""
    import shutil
    base = Path(base)
    assert base.is_absolute() and not base.is_symlink() and base.resolve() == base
    matches = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        fields, filesystem = line.split(' - ', 1)
        if fields.split()[4] == '/dev/shm':
            matches.append(filesystem.split()[0])
    assert len(matches) == 1
    return dict(base=str(base), filesystem=matches[0], mount='/dev/shm',
        device=os.stat(base).st_dev, mount_device=os.stat('/dev/shm').st_dev, root_device=os.stat('/').st_dev,
        free=shutil.disk_usage(base).free, root_free=shutil.disk_usage('/').free)


def verify_schedule(runs):
    assert len(runs) == 4
    assert [(r['phase'], r['family'], r['engine']) for r in runs] == [('timing','whisper',engine) for engine in ENGINES]
    identities = [(r['child']['pid'], r['child']['birth']) for r in runs]
    assert len(set(identities)) == 4, 'Timing processes must be fresh'


def replace_pending(text, section):
    """Refuse stale or ambiguous document layouts without changing another table."""
    assert text.count(PENDING) == text.count(NEXT) == 1
    start = text.index(PENDING); end = text.index(NEXT, start)
    assert text[start:end].count('### ') == 0 and section.endswith('\n\n')
    assert '### Audio: matched AMD Whisper baseline' not in text
    return text[:start]+section+text[end:]
