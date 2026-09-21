"""Keep original public/native gates; compare allocation to the request-context predecessor."""
from applications import *

AUDITOR = ROOT / 'tests/pyannote/request-contexts/audit.py'


def load():
    source = AUDITOR.read_text(encoding='utf8')
    def replace(old, new):
        nonlocal source
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    replace("for name in ['builds.json', 'qualification.json']:", "for name in ['qualification.json']:")
    replace("prior_rows = [r for worker in ['0-candidate', '3-candidate'] for r in read(PUBLIC / 'process' / worker / 'output/result.json')['records']]",
            "prior_rows = read(PRIOR / 'dialogue-output/result.json')['records']")
    replace("old_meetings = read(MEETINGS / 'output-run/result.json');", "old_meetings = read(PRIOR / 'meetings-run-output/result.json');")
    replace("read(BASE / 'instructions.json')", "read(MODEL / 'instructions.json')")
    namespace = dict(globals(), __name__='original_application_auditor', __file__=str(Path(__file__)))
    exec(compile(source, str(AUDITOR), 'exec'), namespace)
    return namespace


if __name__ == '__main__':
    prepared = read(BASE / 'applications-prepared.json')
    assert prepared['files'][rel(AUDITOR)] == pin(AUDITOR)
    load()['main']()
