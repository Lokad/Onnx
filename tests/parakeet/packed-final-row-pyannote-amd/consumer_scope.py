"""Preserve every Pyannote numerical/public assertion and worker contract."""
from pathlib import Path
from protocol import pin

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[2]
ORIGINAL = TOOLS.parent/'observed-dense-where-pyannote-amd'


def verify_scope():
    files = {}
    for name in ['protocol.py', 'checks.py', 'remote.py', 'candidate_protocol.py',
                 'qualify_outputs.py', 'test_semantics.py', 'audit.py']:
        before = ORIGINAL/name
        expected = before.read_bytes()
        if name == 'audit.py':
            for old,new in [(b'M70 current release Core37c24375/Data cc37b19e',b'M78 qualified release Coref95a13c5/Dataa893952f'),
                (b'M70 observed-mask Coref95a13c5/Dataa893952f',b'M78 packed final row Core49901366/Data01e9e784')]:
                assert expected.count(old)==1;expected=expected.replace(old,new)
            assert (TOOLS/name).read_text(encoding='utf8')==expected.decode().replace('\r\n','\n'),name
        else:assert (TOOLS/name).read_bytes()==expected,name
        files[before.relative_to(ROOT).as_posix()] = pin(before)
    return files


if __name__ == '__main__':
    print(verify_scope())
