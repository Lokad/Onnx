"""Keep shared-model numerical checks and staging behavior unchanged."""
import ast
from pathlib import Path
from protocol import pin

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[2]
PARENT = TOOLS.parent/'owned-batch-isolation-shared-amd'


def verify_scope():
    files = {}
    for name in ['protocol.py','checks.py','remote.py','audit.py']:
        assert (TOOLS/name).read_bytes()==(PARENT/name).read_bytes(),name
        files[(PARENT/name).relative_to(ROOT).as_posix()]=pin(PARENT/name)
    expected=(PARENT/'remote_prepare.py').read_text()
    for old in ['parakeet-slice-dense-conversion-models-20260925','parakeet-owned-batch-isolation-models-20260925']:
        expected=expected.replace(old,'parakeet-pad-current-models-20260926')
    expected=expected.replace('parakeet-owned-batch-isolation-release-app-20260925','parakeet-pad-current-app-20260926')
    assert (TOOLS/'remote_prepare.py').read_text()==expected
    def body(path):
        text=path.read_text()
        return next(ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='prepare')
    expected=body(PARENT/'prepare.py')
    marker="    copy(ROOT/'tests/parakeet/reduction-shared/qualify_v2.py', bundle/'evidence/original-auditor.py')"
    extra="    copy(MODELS/'collected/evidence/compatibility.json', bundle/'evidence/compiled-compatibility.json')\n    for name in ['closed.json','analysis.json']: copy(PREVIOUS/name, bundle/'evidence'/('previous-shared-'+name))\n"
    assert body(TOOLS/'prepare.py')==expected.replace(marker,extra+marker)
    for name in ['prepare.py','remote_prepare.py','run.py']:
        files[(PARENT/name).relative_to(ROOT).as_posix()]=pin(PARENT/name)
    return files


if __name__=='__main__': print(verify_scope())
