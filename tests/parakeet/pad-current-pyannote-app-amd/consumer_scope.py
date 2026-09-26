"""Preserve all requests, output checks, scoring gates and process accounting."""
from pathlib import Path
from protocol import pin

TOOLS=Path(__file__).resolve().parent
ROOT=TOOLS.parents[2]
PARENT=TOOLS.parent/'owned-batch-isolation-pyannote-app-amd'
UNCHANGED=['protocol.py','admission.py','semantics.py','meeting_protocol.py','meetings_audit.py',
           'remote.py','checks.py','test_admission.py','test_semantics.py']


def verify_scope():
    files={}
    for name in UNCHANGED:
        assert (TOOLS/name).read_bytes()==(PARENT/name).read_bytes(),name
    for name in ['audit.py','run.py']:
        expected=(PARENT/name).read_text(encoding='utf8')
        expected=expected.replace('Qualified release Coref95a13c5/Dataa893952f','Qualified current root')
        expected=expected.replace('Dispatch relocation Coree07a4518/Data01e9e784','Current-root padding dispatcher')
        expected=expected.replace('parakeet-owned-batch-isolation-pyannote-app-amd-20260925','parakeet-pad-current-pyannote-app-amd-20260926')
        expected=expected.replace('parakeet-owned-batch-isolation-pyannote-app-20260925','parakeet-pad-current-pyannote-app-20260926')
        assert (TOOLS/name).read_text(encoding='utf8')==expected,name
    expected=(PARENT/'remote_prepare.py').read_text(encoding='utf8')
    for before,after in [
        ('parakeet-owned-batch-isolation-pyannote-20260925','parakeet-pad-current-pyannote-20260926'),
        ('parakeet-owned-batch-isolation-models-20260925','parakeet-pad-current-models-20260926'),
        ('parakeet-owned-batch-isolation-shared-20260925','parakeet-pad-current-shared-20260926'),
        ('parakeet-owned-batch-isolation-release-app-20260925','parakeet-pad-current-app-20260926'),
        ("PRIOR['parakeet-release'] = Path('/dev/shm/lokad-parakeet-slice-dense-conversion-models-20260925')",
         "PRIOR['graphs'] = Path('/dev/shm/lokad-parakeet-pad-current-graphs-v2-20260926')"),
        ("pin(BASE/'evidence'/label/name)","pin(BASE/'evidence'/('graph-qualification' if label=='graphs' else label)/name)"),
        ("previous_owner=read(MODELS/'collection.json')['identities'][0]","previous_owner=read(PRIOR['graphs']/'collection.json')['identities'][0]"),
        ('Qualified release Coref95a13c5/Dataa893952f','Qualified current root'),
        ('Dispatch relocation Coree07a4518/Data01e9e784','Current-root padding dispatcher')]:
        assert before in expected,before;expected=expected.replace(before,after)
    assert (TOOLS/'remote_prepare.py').read_text(encoding='utf8')==expected
    for name in [*UNCHANGED,'audit.py','run.py','remote_prepare.py']:
        files[(PARENT/name).relative_to(ROOT).as_posix()]=pin(PARENT/name)
    return files


if __name__=='__main__':print(verify_scope())
