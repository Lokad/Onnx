"""Preserve all application, meeting, numerical and exact-clock contracts."""
from pathlib import Path
from protocol import pin

TOOLS=Path(__file__).resolve().parent
ROOT=TOOLS.parents[2]
ORIGINAL=TOOLS.parent/'first-use-kernels-pyannote-app-amd'


def verify_scope():
    files={}
    for name in ['protocol.py','checks.py','admission.py','semantics.py',
                 'meeting_protocol.py','meetings_audit.py','remote.py',
                 'test_admission.py','test_semantics.py','audit.py']:
        source=ORIGINAL/name;expected=source.read_bytes()
        if name=='audit.py':
            before=b'M43 first-use float kernels Corec00a25b4/Data9b623be9'
            assert expected.count(before)==1
            expected=expected.replace(before,b'M54 wide-entry first-use Core672e5f30/Data065b7a7f')
        assert (TOOLS/name).read_bytes()==expected,name
        files[source.relative_to(ROOT).as_posix()]=pin(source)
    return files


if __name__=='__main__':print(verify_scope())
