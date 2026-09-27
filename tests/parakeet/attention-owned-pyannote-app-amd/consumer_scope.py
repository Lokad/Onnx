"""Preserve original application workers, numerical checks and exact scorers."""
import ast
import hashlib
import json
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[2]
PARENT = TOOLS.parent/'pad-current-pyannote-app-amd'
PREVIOUS = TOOLS.parent/'decoder-lstm-layout-pyannote-app-amd'


def pin(path): return dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def remote_preparation():
    source = (PREVIOUS/'remote_prepare.py').read_text()
    for case in ['pyannote', 'models', 'shared', 'app', 'graphs']:
        before = '/dev/shm/lokad-lstmlayout-'+case+'-20260927'
        assert source.count(before) == 1, before
        source = source.replace(before, '/dev/shm/lokad-attention-owned-'+case+'-20260928')
    assert source.count('Prepared LSTM layout candidate') == 1
    source = source.replace('Prepared LSTM layout candidate', 'Owned attention candidate')
    assert source.count('copy_function=os.link') == 2
    assert source.count('os.link(source, folder/name)') == 1
    source = source.replace('copy_function=os.link', 'copy_function=link_retained')
    source = source.replace('os.link(source, folder/name)', 'link_retained(source, folder/name)')
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


def verify_scope():
    frozen = json.loads((ROOT/'artifacts/parakeet-pad-current-pyannote-app-amd-20260926/prepared.json').read_text())['files']
    files = {}
    for path in PARENT.iterdir():
        if path.is_file():
            name = path.relative_to(ROOT).as_posix()
            assert pin(path) == frozen[name], name
            files[name] = pin(path)
    closed = ROOT/'artifacts/parakeet-decoder-lstm-layout-pyannote-app-amd-20260927'
    proof = json.loads((closed/'closed.json').read_text())
    assert pin(closed/'prepared.json') == proof['files']['prepared.json']
    bindings = json.loads((closed/'prepared.json').read_text())['files']
    for name in ['remote_prepare.py', 'prepare.py', 'prerequisites.py', 'audit.py', 'run.py', 'test_prerequisites.py']:
        path = PREVIOUS/name; relative = path.relative_to(ROOT).as_posix()
        assert pin(path) == bindings[relative], name
        files[relative] = pin(path)
    assert (TOOLS/'remote_prepare.py').read_text() == remote_preparation()
    assert (TOOLS/'test_prerequisites.py').read_text() == (PREVIOUS/'test_prerequisites.py').read_text().replace(
        'test_erased_component_failure', 'test_wrong_census').replace(
        "args[2]['failed_component_controls'].pop()", "args[2]['actual_model_census']['sha256'] = '0'*64").replace(
        'erased component failures', 'wrong census provenance')
    assert (TOOLS/'audit.py').read_text() == (PREVIOUS/'audit.py').read_text().replace(
        'Prepared LSTM layout candidate', 'Owned attention candidate')
    def body(path, name):
        source = path.read_text()
        return next(ast.get_source_segment(source, node) for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    assert body(TOOLS/'prepare.py', 'prepare') == body(PREVIOUS/'prepare.py', 'prepare')
    assert body(TOOLS/'prerequisites.py', 'verify_comparisons') == body(PREVIOUS/'prerequisites.py', 'verify_comparisons')
    def model_checks(path):
        source = path.read_text()
        return source.split('    for role in ROLES:', 1)[1].split("    assert compatible['passed']", 1)[0]
    assert model_checks(TOOLS/'prerequisites.py') == model_checks(PREVIOUS/'prerequisites.py')
    assert not any((TOOLS/name).exists() for name in ['remote.py', 'protocol.py', 'checks.py', 'admission.py', 'semantics.py'])
    return files


if __name__ == '__main__': print(json.dumps(dict(passed=True, frozen_files=len(verify_scope()))))
