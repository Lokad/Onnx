"""Preserve original root jobs, complete test census and package qualification."""
import ast
from source_scope import ROOT,TOOLS,PARENT,APP,load
from protocol import pin,read

PREVIOUS=TOOLS.parent/'attention-owned-root-amd'


def body(path,name):
    source=path.read_text()
    return next(ast.get_source_segment(source,node) for node in ast.parse(source).body
                if isinstance(node,ast.FunctionDef) and node.name==name)


def remote_preparation():
    source=(PREVIOUS/'remote_prepare.py').read_text()
    for case in ['pyannote','models','shared','app','graphs','pyannote-app']:
        before='/dev/shm/lokad-attention-owned-'+case+'-20260928'
        assert before in source
        source=source.replace(before,'/dev/shm/lokad-transpose-axis-'+case+'-20260928')
    for before,after in [
        ('/dev/shm/lokad-pwt-pyannote-app-20260927','/dev/shm/lokad-attention-owned-pyannote-app-20260928'),
        ('/dev/shm/lokad-pwt-root-20260927','/dev/shm/lokad-attention-owned-root-recovery-20260928')]:
        assert source.count(before)==1
        source=source.replace(before,after)
    return source


def verify_scope():
    original=read(ROOT/'artifacts/parakeet-pad-current-root-amd-20260926/prepared.json')['files']
    files={}
    for path in PARENT.iterdir():
        if path.is_file():
            name=path.relative_to(ROOT).as_posix();assert pin(path)==original[name],name
            files[name]=pin(path)
    qualified=ROOT/'artifacts/parakeet-attention-owned-root-recovery-amd-20260928'
    assert pin(qualified/'closed.json')['sha256']=='ee4a38ff671cc3fd8cd608c0dc3008f5c1b99f61ba6c139f3deeaf0e9039305e'
    prior=read(qualified/'closed.json')['local_inputs']
    for name in ['prepare.py','checks.py','remote_prepare.py','prerequisites.py','warning_census.py','audit.py']:
        path=PREVIOUS/name;relative=path.relative_to(ROOT).as_posix()
        assert pin(path)==prior[relative],name
        files[relative]=pin(path)
    application=load('transpose_application_scope',APP/'consumer_scope.py')
    files.update(application.verify_scope())
    files.update({p.relative_to(ROOT).as_posix():pin(p) for p in APP.iterdir() if p.is_file()})
    assert (TOOLS/'checks.py').read_text()==(PREVIOUS/'checks.py').read_text().replace(
        '(3597,43)','(3603,43)').replace('(3507,133)','(3513,133)')
    assert (TOOLS/'warning_census.py').read_text()==(PREVIOUS/'warning_census.py').read_text().replace(
        'parakeet-pointwise-tail-root-amd-20260927','parakeet-attention-owned-root-recovery-amd-20260928')
    assert (TOOLS/'remote_prepare.py').read_text()==remote_preparation()
    assert (TOOLS/'audit.py').read_bytes()==(PREVIOUS/'audit.py').read_bytes()
    expected=body(PREVIOUS/'prepare.py','prepare')
    for before,after in [
        ("    for name in ['closed.json','analysis.json']:copy(CENSUS/name,'evidence/census/'+name)\n",''),
        ("copy(FIXTURE,'evidence/OwnedAttentionPreparationTests.cs')","copy(FIXTURE,'evidence/TransposeAxisMovementTests.cs')"),
        ('446 root inputs; all3288 Core/697 Data methods and metadata equal the measured attention policy; 29 passing preparation cases and one exact disabled-FMA skip in both FMA-capable suites; source policy unchanged.',
         '447 root inputs; all3288 Core/697 Data methods and metadata equal the measured transpose candidate; six portable layout/bit/ownership facts pass in both full modes; original source policy unchanged.'),
        ('source_files=446','source_files=447')]:
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
