"""Preserve every Pyannote numerical/public assertion and worker contract."""
from pathlib import Path
from protocol import pin

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[2]
ORIGINAL = TOOLS.parent/'first-use-kernels-pyannote-amd'


def verify_scope():
    files = {}
    for name in ['protocol.py', 'checks.py', 'remote.py', 'candidate_protocol.py',
                 'qualify_outputs.py', 'test_semantics.py', 'audit.py']:
        before = ORIGINAL/name
        expected = before.read_bytes()
        if name == 'audit.py':
            old = b'M43 first-use float kernels Corec00a25b4/Data9b623be9'
            assert expected.count(old) == 1
            expected = expected.replace(old, b'M66 composition Core37c24375/Data cc37b19e')
        if name=='audit.py':
            assert expected.count(b'selected M34 Core521bae17/Dataf3b9aa81')==1
            expected=expected.replace(b'selected M34 Core521bae17/Dataf3b9aa81',b'M66 selected Core672e5f30/Data065b7a7f')
        if name=='protocol.py':
            assert expected.count(b'preflight_available=12*GIB')==1
            expected=expected.replace(b'preflight_available=12*GIB',b'preflight_available=11*GIB')
        assert (TOOLS/name).read_bytes() == expected, name
        files[before.relative_to(ROOT).as_posix()] = pin(before)
    return files


if __name__ == '__main__':
    print(verify_scope())
