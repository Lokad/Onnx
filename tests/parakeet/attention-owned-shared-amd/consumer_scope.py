"""Keep the established shared/e5 worker, fixtures and complete audit unchanged."""
import ast
from pathlib import Path
from protocol import pin, read

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[2]
PARENT = TOOLS.parent/'decoder-lstm-layout-shared-amd'


def verify_scope():
    files = {}
    for name in ['protocol.py', 'checks.py', 'remote.py', 'audit.py']:
        assert (TOOLS/name).read_bytes() == (PARENT/name).read_bytes(), name
    expected = (PARENT/'remote_prepare.py').read_text()
    for role in ['models', 'app']:
        expected = expected.replace('lokad-lstmlayout-'+role+'-20260927', 'lokad-attention-owned-'+role+'-20260928')
    assert (TOOLS/'remote_prepare.py').read_text() == expected
    def body(path):
        source = path.read_text()
        return next(ast.get_source_segment(source, node) for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'prepare')
    assert body(TOOLS/'prepare.py') == body(PARENT/'prepare.py')
    for name in ['protocol.py', 'checks.py', 'remote.py', 'audit.py', 'prepare.py', 'remote_prepare.py', 'run.py']:
        files[(PARENT/name).relative_to(ROOT).as_posix()] = pin(PARENT/name)
    transport = TOOLS.parent/'owned-batch-isolation-shared-amd/run.py'
    name = transport.relative_to(ROOT).as_posix()
    assert pin(transport) == read(ROOT/'artifacts/parakeet-pad-current-shared-amd-20260926/prepared.json')['files'][name]
    files[name] = pin(transport)
    return files


if __name__ == '__main__': print(verify_scope())
