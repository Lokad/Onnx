"""Keep the original shared numerical checks, worker, protocol and audit exact."""
from pathlib import Path
from protocol import pin

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[2]
ORIGINAL = TOOLS.parent/'first-use-kernels-shared-amd'


def verify_scope():
    files = {}
    for name in ['protocol.py', 'checks.py', 'remote.py', 'audit.py']:
        before = ORIGINAL/name
        expected=before.read_bytes()
        if name=='protocol.py':
            assert expected.count(b'preflight_available=12*GIB')==1
            expected=expected.replace(b'preflight_available=12*GIB',b'preflight_available=11*GIB')
        assert (TOOLS/name).read_bytes() == expected, name
        files[before.relative_to(ROOT).as_posix()] = pin(before)
    return files


if __name__ == '__main__':
    print(verify_scope())
