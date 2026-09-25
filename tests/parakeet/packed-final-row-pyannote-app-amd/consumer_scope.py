"""Preserve all scoring and public checks; bind only the product lineage."""
from pathlib import Path
from protocol import pin
from adaptation_scope import with_lineage
TOOLS=Path(__file__).resolve().parent
ROOT=TOOLS.parents[2]
ORIGINAL=TOOLS.parent/'slice-dense-conversion-pyannote-app-amd'
UNCHANGED=['protocol.py','admission.py','semantics.py','meeting_protocol.py',
           'meetings_audit.py','remote.py','test_admission.py','test_semantics.py','graph_prerequisite.py']


def verify_scope():
    files={}
    for name in [*UNCHANGED,'checks.py','audit.py']:
        source=ORIGINAL/name
        if name in UNCHANGED:
            assert (TOOLS/name).read_bytes()==source.read_bytes(),name
        else:
            expected=source.read_text(encoding='utf8')
            if name=='checks.py':expected=with_lineage(expected)
            else:
                for before,after in [('M73 current release Coref95a13c5/Dataa893952f','M78 current release Coref95a13c5/Dataa893952f'),
                                     ('M73 slice conversion Core49c3a958/Dataa893952f','M78 packed final row Core49901366/Data01e9e784')]:
                    assert expected.count(before)==1;expected=expected.replace(before,after)
            assert (TOOLS/name).read_text(encoding='utf8')==expected,name
        files[source.relative_to(ROOT).as_posix()]=pin(source)
    return files


if __name__=='__main__':print(verify_scope())
