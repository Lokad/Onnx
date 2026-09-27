"""Retain the original exact shared-model checks and fixture preparation."""
import ast
from pathlib import Path
from protocol import pin, read

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[2]
PARENT = TOOLS.parent/'pad-current-shared-amd'


def verify_scope():
    files = {}
    for name in ['protocol.py', 'checks.py', 'remote.py', 'audit.py']:
        assert (TOOLS/name).read_bytes() == (PARENT/name).read_bytes(), name
    expected = (PARENT/'remote_prepare.py').read_text()
    for role in ['models', 'app']:
        expected = expected.replace('parakeet-pad-current-'+role+'-20260926',
                                    'parakeet-decoder-packed-row-'+role+'-20260927')
    assert (TOOLS/'remote_prepare.py').read_text() == expected
    def body(path):
        text = path.read_text()
        return next(ast.get_source_segment(text, n) for n in ast.parse(text).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'prepare')
    assert body(TOOLS/'prepare.py') == body(PARENT/'prepare.py')
    for name in ['protocol.py', 'checks.py', 'remote.py', 'audit.py', 'prepare.py', 'remote_prepare.py', 'run.py']:
        files[(PARENT/name).relative_to(ROOT).as_posix()] = pin(PARENT/name)
    transport = PARENT.parent/'owned-batch-isolation-shared-amd/run.py'
    name = transport.relative_to(ROOT).as_posix()
    assert pin(transport) == read(ROOT/'artifacts/parakeet-pad-current-shared-amd-20260926/prepared.json')['files'][name]
    files[name] = pin(transport)
    return files


if __name__ == '__main__': print(verify_scope())
