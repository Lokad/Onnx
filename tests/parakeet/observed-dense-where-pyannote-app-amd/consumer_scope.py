"""Preserve all application, meeting, numerical and exact-clock contracts."""
import ast
from pathlib import Path
from protocol import pin

TOOLS=Path(__file__).resolve().parent
ROOT=TOOLS.parents[2]
ORIGINAL=TOOLS.parent/'validated-composition-pyannote-app-amd'


def verify_scope():
    files={}
    for name in ['protocol.py','checks.py','admission.py','semantics.py',
                 'meeting_protocol.py','meetings_audit.py','remote.py',
                 'test_admission.py','test_semantics.py','audit.py']:
        source=ORIGINAL/name;expected=source.read_bytes()
        if name=='audit.py':
            for old,new in [(b'M66 selected Core672e5f30/Data065b7a7f',b'M70 current release Core37c24375/Data cc37b19e'),
                (b'M66 composition Core37c24375/Data cc37b19e',b'M70 observed-mask Coref95a13c5/Dataa893952f')]:
                assert expected.count(old)==1;expected=expected.replace(old,new)
            assert (TOOLS/name).read_text(encoding='utf8')==expected.decode().replace('\r\n','\n'),name
        else:assert (TOOLS/name).read_bytes()==expected,name
        files[source.relative_to(ROOT).as_posix()]=pin(source)
    return files


if __name__=='__main__':print(verify_scope())
