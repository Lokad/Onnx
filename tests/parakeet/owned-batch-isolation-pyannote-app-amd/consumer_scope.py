"""Preserve execution, output validation and scores; update only prerequisites."""
from pathlib import Path
from protocol import pin

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[2]
ORIGINAL = TOOLS.parent/'packed-final-row-pyannote-app-amd'
UNCHANGED = ['protocol.py', 'admission.py', 'semantics.py', 'meeting_protocol.py',
             'meetings_audit.py', 'remote.py', 'test_admission.py', 'test_semantics.py']


def expected(name):
    value = (ORIGINAL/name).read_text(encoding='utf8')
    if name == 'checks.py':
        value = value.replace('from lineage import qualify_lineage\n', '')
        start, end = value.index('def prereqs(base, spec):'), value.index('\ndef records_protocol(base):')
        value = value[:start]+'''def prereqs(base, spec):
    from prerequisites import verify
    return verify(base, spec)

'''+value[end:]
    elif name == 'audit.py':
        value = value.replace('M78 current release Coref95a13c5/Dataa893952f', 'Qualified release Coref95a13c5/Dataa893952f')
        value = value.replace('M78 packed final row Core49901366/Data01e9e784', 'Dispatch relocation Coree07a4518/Data01e9e784')
    elif name == 'run.py':
        value = value.replace('packed-final-row-pyannote-app', 'owned-batch-isolation-pyannote-app')
        value = value.replace('def observe():\n', "def observe():\n    assert not (BASE/'closed.json').exists(), 'Preserve the closed campaign'\n")
    return value


def verify_scope():
    files = {}
    for name in [*UNCHANGED, 'checks.py', 'audit.py', 'run.py']:
        source = ORIGINAL/name
        if name in UNCHANGED:
            assert (TOOLS/name).read_bytes() == source.read_bytes(), name
        else:
            assert (TOOLS/name).read_text(encoding='utf8') == expected(name), name
        files[source.relative_to(ROOT).as_posix()] = pin(source)
    return files


if __name__ == '__main__':
    print(verify_scope())
