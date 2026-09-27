"""Reconcile the adapter exactly to the closed padding shared-model lane."""
import ast
from pathlib import Path
from protocol import pin, read

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[2]
PARENT = TOOLS.parent/'pad-current-shared-amd'


def verify_scope():
    files = {}
    for name in ['protocol.py', 'remote.py', 'audit.py']:
        assert (TOOLS/name).read_bytes() == (PARENT/name).read_bytes(), name
    expected = (PARENT/'checks.py').read_text().replace('and exact same-platform bits', 'with recorded arithmetic differences')
    expected = expected.replace('from protocol import pin, read', 'from protocol import pin, read\nfrom cross_numeric import compare')
    expected = expected.replace('        if selected is not None:\n', '        comparison = None\n        if selected is not None:\n')
    expected = expected.replace("            assert path.read_bytes() == (base/('selected-'+mode)/'output'/other['file']).read_bytes()",
        "            comparison = compare(path, base/('selected-'+mode)/'output'/other['file'])")
    expected = expected.replace('exact_selected=True if selected is not None else None',
        "exact_selected=None if comparison is None else comparison['bit_identical'], selected_comparison=comparison")
    assert (TOOLS/'checks.py').read_text() == expected
    expected = (PARENT/'remote_prepare.py').read_text()
    for role in ['models', 'app']:
        expected = expected.replace('parakeet-pad-current-'+role+'-20260926', 'parakeet-rational-sigmoid-'+role+'-20260927')
    expected = expected.replace('exact selected/native outputs; no performance score.',
        'original native bounds and bounded selected differences; no performance score.')
    assert (TOOLS/'remote_prepare.py').read_text() == expected
    def body(path):
        text = path.read_text()
        return next(ast.get_source_segment(text, n) for n in ast.parse(text).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'prepare')
    expected = body(PARENT/'prepare.py').replace(
        "['protocol.py', 'remote.py', 'remote_prepare.py', 'checks.py']",
        "['protocol.py', 'remote.py', 'remote_prepare.py', 'checks.py', 'cross_numeric.py']")
    assert body(TOOLS/'prepare.py') == expected
    for name in ['protocol.py', 'checks.py', 'remote.py', 'audit.py', 'prepare.py', 'remote_prepare.py', 'run.py']:
        files[(PARENT/name).relative_to(ROOT).as_posix()] = pin(PARENT/name)
    transport = PARENT.parent/'owned-batch-isolation-shared-amd/run.py'
    name = transport.relative_to(ROOT).as_posix()
    assert pin(transport) == read(ROOT/'artifacts/parakeet-pad-current-shared-amd-20260926/prepared.json')['files'][name]
    files[name] = pin(transport)
    return files


if __name__ == '__main__': print(verify_scope())
