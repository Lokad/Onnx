"""Keep all request validation, scoring, controls and resource bounds intact."""
import ast
from pathlib import Path


def verify_scope():
    here=Path(__file__).resolve().parent;prior=here.parent/'observed-dense-where-app-amd'
    def methods(path):
        source=path.read_text()
        return {node.name:ast.get_source_segment(source,node) for node in ast.parse(source).body if isinstance(node,ast.FunctionDef)}
    before,after=methods(prior/'checks.py'),methods(here/'checks.py')
    assert before.keys()==after.keys()
    assert [name for name in before if before[name]!=after[name]]==['prereqs']
    for name in ['remote.py','statistics_exact.py','test_admission.py','audit.py']:
        assert (here/name).read_bytes()==(prior/name).read_bytes(),name
    assert (here/'protocol.py').read_bytes()==(prior/'protocol.py').read_bytes()
    return True


if __name__=='__main__':print(verify_scope())
