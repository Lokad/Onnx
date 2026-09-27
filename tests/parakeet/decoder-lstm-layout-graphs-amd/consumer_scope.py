"""Freeze inherited graph workers, numerical checks, exact scorers and auditor."""
import hashlib
import json
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[2]
PARENT = TOOLS.parent/'pad-current-graphs-amd'
STAGING = TOOLS.parent/'pad-current-graphs-v2-amd'


def pin(path): return dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def verify_scope():
    prepared = ROOT/'artifacts/parakeet-pad-current-graphs-v2-amd-20260926/prepared.json'
    original = json.loads(prepared.read_text())['files']; files = {}
    for path in PARENT.iterdir():
        if path.is_file():
            name = path.relative_to(ROOT).as_posix()
            assert pin(path) == original[name], name
            files[name] = pin(path)
    staging = STAGING/'remote_prepare.py'
    assert pin(staging) == original[staging.relative_to(ROOT).as_posix()]
    files[staging.relative_to(ROOT).as_posix()] = pin(staging)
    expected = staging.read_text()
    block = "    failure=BASE/'evidence/launch-failure/collection.json'\n" \
        "    assert pin(failure)==pin(Path('/dev/shm/lokad-parakeet-pad-current-graphs-20260926/collection.json'))\n" \
        "    assert read(failure)['terminal'] and read(failure)['code']==1\n" \
        "    assert not any(live(i) for i in read(failure)['identities'])\n"
    assert expected.count(block) == 1
    assert (TOOLS/'remote_prepare.py').read_text() == expected.replace(block, '')
    assert not any((TOOLS/n).exists() for n in ['remote.py', 'checks.py', 'statistics.py'])
    return files


if __name__ == '__main__': print(verify_scope())
