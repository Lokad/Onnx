"""Preserve all application, meeting, numerical and exact-clock contracts."""
import ast
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
            expected=expected.replace(before,b'M66 composition Core37c24375/Data cc37b19e')
        if name=='audit.py':
            assert expected.count(b'selected M34 Core521bae17/Dataf3b9aa81')==1
            expected=expected.replace(b'selected M34 Core521bae17/Dataf3b9aa81',b'M66 selected Core672e5f30/Data065b7a7f')
        if name=='protocol.py':
            assert expected.count(b'preflight_available=12*GIB')==1
            expected=expected.replace(b'preflight_available=12*GIB',b'preflight_available=11*GIB')
        if name=='checks.py':
            def without_prereqs(data):
                tree=ast.parse(data)
                assert sum(isinstance(n,ast.FunctionDef) and n.name=='prereqs' for n in tree.body)==1
                tree.body=[n for n in tree.body if not(isinstance(n,ast.FunctionDef) and n.name=='prereqs')]
                return ast.dump(tree,include_attributes=False)
            assert without_prereqs((TOOLS/name).read_bytes())==without_prereqs(expected),name
        else:assert (TOOLS/name).read_bytes()==expected,name
        files[source.relative_to(ROOT).as_posix()]=pin(source)
    return files


if __name__=='__main__':print(verify_scope())
