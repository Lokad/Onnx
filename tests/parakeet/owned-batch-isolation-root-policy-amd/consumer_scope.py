"""Keep full numerical/suite/package checks, with one explicit metadata delta."""
import ast
from pathlib import Path
from protocol import JOBS,pin
from source_scope import ROOT,TOOLS,ORIGINAL,GENERATOR,load,verify_source,root_files


def functions(path):
    text=path.read_text(encoding='utf8')
    return {n.name:ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}


def verify_scope():
    original=load('retained_root_consumer_scope',ORIGINAL/'consumer_scope.py')
    files=original.verify_scope()
    for name in ['admission.py','new_cases.py','graph_prerequisite.py','test_census.py','test_guards.py']:
        assert (TOOLS/name).read_bytes()==(ORIGINAL/name).read_bytes(),name
    expected=(ORIGINAL/'remote_prepare.py').read_text(encoding='utf8').replace(
        'all methodbodies/implementationflags/API equal to measured relocation',
        'all methodbodies/implementationflags equal to measured relocation; only preparation visibility and Data friend metadata change')
    assert (TOOLS/'remote_prepare.py').read_text(encoding='utf8')==expected
    expected=(ORIGINAL/'protocol.py').read_text(encoding='utf8').replace("'tensors-build','inventory'","'tensors-build','bridge-restore','bridge-build','inventory'")
    assert (TOOLS/'protocol.py').read_text(encoding='utf8')==expected and len(JOBS)==18
    before,after=functions(ORIGINAL/'checks.py'),functions(TOOLS/'checks.py')
    assert set(after)==set(before)|{'metadata_delta'}
    for name in before:
        if name!='inventory':assert before[name]==after[name],name
    before,after=functions(ORIGINAL/'remote.py'),functions(TOOLS/'remote.py')
    assert before.keys()==after.keys()
    assert {name for name in before if before[name]!=after[name]}=={'command_for','after'}
    expected=(ORIGINAL/'run.py').read_text(encoding='utf8').replace('owned-batch-isolation-root-amd-20260925','owned-batch-isolation-root-policy-amd-20260925').replace(
        "'/dev/shm/lokad-parakeet-owned-batch-isolation-root-20260925'", "'/dev/shm/lokad-parakeet-owned-batch-isolation-root-policy-20260925'")
    assert (TOOLS/'run.py').read_text(encoding='utf8')==expected
    source=verify_source();assert len(root_files(source))==435
    for folder in [ORIGINAL,TOOLS.parent/'owned-batch-isolation-root-correction']:
        for path in folder.iterdir():
            if path.is_file():files[path.relative_to(ROOT).as_posix()]=pin(path)
    for name in ['owned-batch-isolation-build/Bridge.cs.txt','selected-profile-build-amd/Bridge.csproj']:
        path=TOOLS.parent/name;files[path.relative_to(ROOT).as_posix()]=pin(path)
    return files


if __name__=='__main__':
    print(dict(passed=True,frozen_helpers=len(verify_scope()),jobs=len(JOBS)))
