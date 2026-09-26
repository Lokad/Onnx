"""Reuse the root worker, transport, package consumer and independent auditor."""
import ast
from protocol import JOBS,pin
from source_scope import ROOT,TOOLS,ORIGINAL,verify_source,root_files


def functions(path):
    text=path.read_text(encoding='utf8')
    return {node.name:ast.get_source_segment(text,node) for node in ast.parse(text).body if isinstance(node,ast.FunctionDef)}


def verify_scope():
    for name in ['remote.py','audit.py']:
        assert (TOOLS/name).read_bytes()==(ORIGINAL/name).read_bytes(),name
    expected=(ORIGINAL/'protocol.py').read_text(encoding='utf8').replace('\nGIB =',"\nROLES = ['selected', 'candidate']\nGIB =")
    assert (TOOLS/'protocol.py').read_text(encoding='utf8')==expected and len(JOBS)==18
    expected=(ORIGINAL/'run.py').read_text(encoding='utf8').replace(
        'owned-batch-isolation-root-policy-amd-20260925','pad-current-root-amd-20260926').replace(
        'owned-batch-isolation-root-policy-20260925','pad-current-root-20260926')
    assert (TOOLS/'run.py').read_text(encoding='utf8')==expected
    expected=(ORIGINAL/'warning_census.py').read_text(encoding='utf8').replace(
        'parakeet-observed-dense-where-root-amd-v2-20260924','parakeet-owned-batch-isolation-root-policy-amd-20260925')
    assert (TOOLS/'warning_census.py').read_text(encoding='utf8')==expected
    before,after=functions(ORIGINAL/'checks.py'),functions(TOOLS/'checks.py')
    assert before.keys()==after.keys()
    assert {k for k in before if before[k]!=after[k]}=={'metadata_delta','inventory','suite','suite256'}
    app=TOOLS.parent/'pad-current-pyannote-app-amd'
    for local,old in [('graph_prerequisite.py','graph_prerequisite.py'),
                      ('model_prerequisites.py','prerequisites.py'),('application_admission.py','admission.py')]:
        assert (TOOLS/local).read_bytes()==(app/old).read_bytes(),local
    assert len(root_files(verify_source()))==437
    paths=[p for folder in [ORIGINAL,app] for p in folder.iterdir() if p.is_file()]
    paths += [TOOLS.parent/'owned-batch-isolation-build/Bridge.cs.txt',
              TOOLS.parent/'selected-profile-build-amd/Bridge.csproj',
              TOOLS.parent/'pad-current-integration/prepare_tests.py']
    return {p.relative_to(ROOT).as_posix():pin(p) for p in paths}


if __name__=='__main__':print(dict(passed=True,frozen_helpers=len(verify_scope()),jobs=len(JOBS)))
