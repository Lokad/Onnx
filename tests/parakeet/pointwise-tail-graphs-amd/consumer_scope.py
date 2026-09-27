"""Freeze original graph workers, numerical checks, exact scorers and auditor."""
import ast
import hashlib
import json
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[2]
PARENT = TOOLS.parent/'pad-current-graphs-amd'
PREVIOUS = TOOLS.parent/'decoder-lstm-layout-graphs-amd'


def pin(path): return dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def remote_preparation():
    source = (PREVIOUS/'remote_prepare.py').read_text()
    assert source.count('import os\n') == 1
    source = source.replace('import os\n', 'import os\nimport shutil\n')
    assert source.count('os.link(original,target)') == 1
    source = source.replace('os.link(original,target)', 'link_retained(original,target)')
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
    prepared = ROOT/'artifacts/parakeet-pad-current-graphs-v2-amd-20260926/prepared.json'
    original = json.loads(prepared.read_text())['files']; files = {}
    for path in PARENT.iterdir():
        if path.is_file():
            name = path.relative_to(ROOT).as_posix()
            assert pin(path) == original[name], name
            files[name] = pin(path)
    closed = ROOT/'artifacts/parakeet-decoder-lstm-layout-graphs-amd-20260927'
    proof = json.loads((closed/'closed.json').read_text())
    assert pin(closed/'prepared.json') == proof['files']['prepared.json']
    frozen = json.loads((closed/'prepared.json').read_text())['files']
    for name in ['remote_prepare.py', 'prepare.py', 'audit.py', 'run.py', 'test_prerequisites.py']:
        path = PREVIOUS/name; relative = path.relative_to(ROOT).as_posix()
        assert pin(path) == frozen[relative], name
        files[relative] = pin(path)
    assert (TOOLS/'remote_prepare.py').read_text() == remote_preparation()
    for name in ['audit.py', 'test_prerequisites.py']:
        assert (TOOLS/name).read_bytes() == (PREVIOUS/name).read_bytes(), name
    def body(path):
        source = path.read_text()
        return next(ast.get_source_segment(source, node) for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'prepare')
    assert body(TOOLS/'prepare.py') == body(PREVIOUS/'prepare.py')
    assert not any((TOOLS/name).exists() for name in ['remote.py', 'checks.py', 'statistics.py'])
    return files


if __name__ == '__main__': print(verify_scope())
