"""Preserve the original root jobs, test census mechanics and package auditor."""
import ast
from source_scope import ROOT,TOOLS,PARENT,APP,load
from protocol import pin,read

PREVIOUS=TOOLS.parent/'pointwise-tail-root-amd'


def body(path,name):
    source=path.read_text()
    return next(ast.get_source_segment(source,node) for node in ast.parse(source).body
                if isinstance(node,ast.FunctionDef) and node.name==name)


def remote_preparation():
    source=(PREVIOUS/'remote_prepare.py').read_text()
    for case in ['pyannote','models','shared','app','graphs','pyannote-app']:
        before='/dev/shm/lokad-pwt-'+case+'-20260927'
        assert before in source
        source=source.replace(before,'/dev/shm/lokad-attention-owned-'+case+'-20260928')
    for case in ['pyannote-app','root']:
        before='/dev/shm/lokad-lstmlayout-'+case+'-20260927'
        assert source.count(before)==1
        source=source.replace(before,'/dev/shm/lokad-pwt-'+case+'-20260927')
    return source


def expected_checks():
    source=(PREVIOUS/'checks.py').read_text()
    source=source.replace('from new_cases import NEW_CASES','from new_cases import NEW_CASES,NEW_SKIPS')
    before="    additions=Counter((test,'Passed') for test in NEW_CASES[name])\n    assert not set(test for test,_ in expected) & set(NEW_CASES[name])"
    after="    additions=Counter((test,'Passed') for test in NEW_CASES[name])\n    additions.update((test,'NotExecuted') for test in NEW_SKIPS[name])\n    assert not set(test for test,_ in expected) & (set(NEW_CASES[name]) | set(NEW_SKIPS[name]))"
    assert source.count(before)==2
    source=source.replace(before,after)
    assert source.count('(3568,42)')==source.count('(3478,132)')==1
    return source.replace('(3568,42)','(3597,43)').replace('(3478,132)','(3507,133)')


def verify_scope():
    original=read(ROOT/'artifacts/parakeet-pad-current-root-amd-20260926/prepared.json')['files']
    files={}
    for path in PARENT.iterdir():
        if path.is_file():
            name=path.relative_to(ROOT).as_posix();assert pin(path)==original[name],name
            files[name]=pin(path)
    prior=read(ROOT/'artifacts/parakeet-pointwise-tail-root-amd-20260927/closed.json')['local_inputs']
    for name in ['prepare.py','checks.py','remote_prepare.py','prerequisites.py','warning_census.py','audit.py']:
        path=PREVIOUS/name;relative=path.relative_to(ROOT).as_posix()
        assert pin(path)==prior[relative],name
        files[relative]=pin(path)
    application=load('attention_application_scope',APP/'consumer_scope.py')
    files.update(application.verify_scope())
    files.update({p.relative_to(ROOT).as_posix():pin(p) for p in APP.iterdir() if p.is_file()})
    assert (TOOLS/'checks.py').read_text()==expected_checks()
    assert (TOOLS/'warning_census.py').read_text()==(PREVIOUS/'warning_census.py').read_text().replace(
        'parakeet-decoder-lstm-layout-root-amd-20260927','parakeet-pointwise-tail-root-amd-20260927')
    assert (TOOLS/'remote_prepare.py').read_text()==remote_preparation()
    assert (TOOLS/'audit.py').read_bytes()==(PREVIOUS/'audit.py').read_bytes()
    expected=body(PREVIOUS/'prepare.py','prepare')
    for before,after in [
        ("    copy(BUILD/'closed.json','evidence/contracts-failure/closed.json')\n    copy(BUILD/'build-review.json','evidence/contracts-build.json')\n    copy(CONTRACTS/'codegen-review.json','evidence/contracts-codegen.json')",
         "    for name in ['closed.json','analysis.json']:copy(CENSUS/name,'evidence/census/'+name)\n    copy(BUILD/'build-review.json','evidence/contracts-build.json')"),
        ("copy(FIXTURE,'evidence/PackedColumnRemainderTests.cs')","copy(FIXTURE,'evidence/OwnedAttentionPreparationTests.cs')"),
        ('445 root inputs; all3288 Core/697 Data methods and metadata equal the measured pointwise candidate; two independent portable remainder facts; source policy unchanged.',
         '446 root inputs; all3288 Core/697 Data methods and metadata equal the measured attention policy; 29 passing preparation cases and one exact disabled-FMA skip in both FMA-capable suites; source policy unchanged.'),
        ('source_files=445','source_files=446')]:
        assert expected.count(before)==1,before
        expected=expected.replace(before,after)
    assert body(TOOLS/'prepare.py','prepare')==expected
    def application_tail(path):
        return path.read_text().split("    app = reports['pyannote-app']",1)[1].split('\n\ndef verify(',1)[0]
    assert application_tail(TOOLS/'prerequisites.py')==application_tail(PREVIOUS/'prerequisites.py')
    for path in [TOOLS.parent/'owned-batch-isolation-build/Bridge.cs.txt',TOOLS.parent/'selected-profile-build-amd/Bridge.csproj']:
        files[path.relative_to(ROOT).as_posix()]=pin(path)
    assert not any((TOOLS/name).exists() for name in ['remote.py','protocol.py','application_admission.py','graph_prerequisite.py'])
    return files


if __name__=='__main__':print(dict(passed=True,frozen_files=len(verify_scope()),jobs=18))
