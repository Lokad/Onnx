"""Keep the complete transcription experiment and its original decisions intact."""
import ast
from pathlib import Path


def verify_scope():
    here=Path(__file__).resolve().parent
    original=here.parent/'observed-dense-where-app-amd'
    for name in ['protocol.py','remote.py','statistics_exact.py','test_admission.py','audit.py']:
        assert (here/name).read_bytes()==(original/name).read_bytes(),name
    def functions(path):
        text=path.read_text()
        return {n.name:ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
    before,after=functions(original/'checks.py'),functions(here/'checks.py')
    assert before.keys()==after.keys()
    assert [n for n in before if before[n]!=after[n]]==['prereqs']
    transport=(here.parent/'packed-final-row-release-app-amd/run.py').read_text()
    transport=transport.replace('packed-final-row-release-app','direct-depthwise-app')
    transport=transport.replace('the M78 full Parakeet comparison','the direct-depthwise full Parakeet comparison')
    assert (here/'run.py').read_text()==transport
    return True


if __name__=='__main__':print(verify_scope())
