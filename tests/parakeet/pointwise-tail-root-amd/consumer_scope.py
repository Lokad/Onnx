"""Keep actual-root workers, complete test comparisons and package audit unchanged."""
import ast
from source_scope import ROOT, TOOLS, PARENT, APP, load
from protocol import pin, read

PREVIOUS = TOOLS.parent/'decoder-lstm-layout-root-amd'


def remote_preparation():
    source = (PREVIOUS/'remote_prepare.py').read_text()
    for case in ['pyannote', 'models', 'shared', 'app', 'graphs', 'pyannote-app']:
        before = '/dev/shm/lokad-lstmlayout-'+case+'-20260927'
        assert before in source, before
        source = source.replace(before, '/dev/shm/lokad-pwt-'+case+'-20260927')
    source = source.replace('/dev/shm/lokad-parakeet-decoder-packed-row-pyannote-app-20260927',
                            '/dev/shm/lokad-lstmlayout-pyannote-app-20260927')
    source = source.replace('/dev/shm/lokad-parakeet-decoder-packed-row-root-20260927',
                            '/dev/shm/lokad-lstmlayout-root-20260927')
    assert source.count('copy_function=os.link') == 1
    source = source.replace('copy_function=os.link', 'copy_function=link_retained')
    marker = '\ndef main():\n'
    assert source.count(marker) == 1
    return source.replace(marker, '''
def link_retained(source, destination):
    source = Path(source).resolve(); destination = Path(destination)
    if source.stat().st_dev == destination.parent.stat().st_dev: os.link(source, destination)
    else: shutil.copy2(source, destination)
    assert pin(source) == pin(destination)
    return str(destination)

def main():
''')


def body(path, name):
    source = path.read_text()
    return next(ast.get_source_segment(source, node) for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def verify_scope():
    original = read(ROOT/'artifacts/parakeet-pad-current-root-amd-20260926/prepared.json')['files']
    files = {}
    for path in PARENT.iterdir():
        if path.is_file():
            name = path.relative_to(ROOT).as_posix()
            assert pin(path) == original[name], name
            files[name] = pin(path)
    prior = read(ROOT/'artifacts/parakeet-decoder-lstm-layout-root-amd-20260927/closed.json')['local_inputs']
    for name in ['prepare.py', 'checks.py', 'remote_prepare.py', 'prerequisites.py', 'warning_census.py', 'audit.py']:
        path = PREVIOUS/name; relative = path.relative_to(ROOT).as_posix()
        assert pin(path) == prior[relative], name
        files[relative] = pin(path)
    application = load('pointwise_application_scope', APP/'consumer_scope.py')
    files.update(application.verify_scope())
    files.update({path.relative_to(ROOT).as_posix():pin(path) for path in APP.iterdir() if path.is_file()})
    expected = (PREVIOUS/'checks.py').read_text().replace('3286','3288').replace('(3566,42)','(3568,42)').replace('(3476,132)','(3478,132)')
    assert (TOOLS/'checks.py').read_text() == expected
    expected = (PREVIOUS/'warning_census.py').read_text().replace(
        'parakeet-decoder-packed-row-root-amd-20260927', 'parakeet-decoder-lstm-layout-root-amd-20260927')
    assert (TOOLS/'warning_census.py').read_text() == expected
    assert (TOOLS/'remote_prepare.py').read_text() == remote_preparation()
    assert (TOOLS/'audit.py').read_bytes() == (PREVIOUS/'audit.py').read_bytes()
    expected = body(PREVIOUS/'prepare.py', 'prepare')
    for before, after in [
        ("copy(SOURCE/'failed.json','evidence/contracts-failure/failed.json')",
         "copy(BUILD/'closed.json','evidence/contracts-failure/closed.json')\n    copy(BUILD/'build-review.json','evidence/contracts-build.json')\n    copy(CONTRACTS/'codegen-review.json','evidence/contracts-codegen.json')"),
        ("SOURCE/'bundle/stage.json'", "SOURCE/'prepared.json'"),
        ("copy(FIXTURE/'prepared.json','evidence/fixture-prepared.json')", "copy(FIXTURE,'evidence/PackedColumnRemainderTests.cs')"),
        ('443 root inputs; all3286 Core/697 Data methods and metadata equal the measured LSTM layout candidate; two unchanged portable projection facts; source policy unchanged.',
         '445 root inputs; all3288 Core/697 Data methods and metadata equal the measured pointwise candidate; two independent portable remainder facts; source policy unchanged.'),
        ('source_files=443','source_files=445')]:
        assert before in expected, before
        expected = expected.replace(before, after)
    assert body(TOOLS/'prepare.py', 'prepare') == expected
    def application_tail(path):
        return path.read_text().split("    app = reports['pyannote-app']",1)[1].split('\n\ndef verify(',1)[0]
    assert application_tail(TOOLS/'prerequisites.py') == application_tail(PREVIOUS/'prerequisites.py')
    for path in [TOOLS.parent/'owned-batch-isolation-build/Bridge.cs.txt',
                 TOOLS.parent/'selected-profile-build-amd/Bridge.csproj']:
        files[path.relative_to(ROOT).as_posix()] = pin(path)
    assert not any((TOOLS/name).exists() for name in ['remote.py','protocol.py','application_admission.py','graph_prerequisite.py'])
    return files


if __name__ == '__main__': print(dict(passed=True, frozen_files=len(verify_scope()), jobs=18))
