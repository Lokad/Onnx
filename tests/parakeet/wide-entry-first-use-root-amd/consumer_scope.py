"""Keep the full worker, test census and package checks unchanged."""
from pathlib import Path
from protocol import pin

TOOLS=Path(__file__).resolve().parent
ROOT=TOOLS.parents[2]
ORIGINAL=TOOLS.parent/'first-use-kernels-root-amd'


def verify_scope():
    files={}
    for name in ['protocol.py','remote.py','audit.py','checks.py']:
        source=ORIGINAL/name;expected=source.read_bytes()
        if name in ['protocol.py','remote.py']:expected=expected.replace(b'M43',b'M54')
        if name=='checks.py':
            assert expected.count(b'3181')==2
            expected=expected.replace(b'3181',b'3189')
        assert (TOOLS/name).read_bytes()==expected,name
        files[source.relative_to(ROOT).as_posix()]=pin(source)
    return files


if __name__=='__main__':print(verify_scope())
