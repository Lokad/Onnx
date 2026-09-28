"""Freeze original graph workers, numerical checks, scorers and preparation."""
import ast
import hashlib
import json
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[2]
PARENT = TOOLS.parent/'pad-current-graphs-amd'
PREVIOUS = TOOLS.parent/'attention-owned-graphs-amd'


def pin(path): return dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def verify_scope():
    prepared = ROOT/'artifacts/parakeet-pad-current-graphs-v2-amd-20260926/prepared.json'
    original = json.loads(prepared.read_text())['files']; files = {}
    for path in PARENT.iterdir():
        if path.is_file():
            name = path.relative_to(ROOT).as_posix()
            assert pin(path) == original[name], name
            files[name] = pin(path)
    closed = ROOT/'artifacts/parakeet-attention-owned-graphs-amd-20260928'
    assert pin(closed/'closed.json')['sha256'] == '0878b709926c77a55eaf82e63194e6d1379e1c877cfd11b7c0ab997e142eb613'
    proof = json.loads((closed/'closed.json').read_text())
    assert pin(closed/'prepared.json') == proof['files']['prepared.json']
    frozen = json.loads((closed/'prepared.json').read_text())['files']
    for name in ['remote_prepare.py', 'prepare.py', 'audit.py', 'run.py', 'test_prerequisites.py']:
        path = PREVIOUS/name; relative = path.relative_to(ROOT).as_posix()
        assert pin(path) == frozen[relative], name
        files[relative] = pin(path)
    for name in ['remote_prepare.py', 'audit.py']:
        assert (TOOLS/name).read_bytes() == (PREVIOUS/name).read_bytes(), name
    assert (TOOLS/'test_prerequisites.py').read_text() == (PREVIOUS/'test_prerequisites.py').read_text().replace(
        'test_wrong_census_is_rejected', 'test_wrong_focused_contract_is_rejected').replace(
        "args[1]['actual_model_census']['sha256']", "args[1]['focused_contracts']['sha256']")
    def body(path):
        source = path.read_text()
        return next(ast.get_source_segment(source, node) for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'prepare')
    assert body(TOOLS/'prepare.py') == body(PREVIOUS/'prepare.py')
    assert not any((TOOLS/name).exists() for name in ['remote.py', 'checks.py', 'statistics.py'])
    return files


if __name__ == '__main__': print(verify_scope())
