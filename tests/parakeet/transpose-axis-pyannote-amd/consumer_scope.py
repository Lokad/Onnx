"""Retain the admitted Pyannote worker, fixtures and complete numerical checks."""
import ast
from pathlib import Path
from protocol import pin, read

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[2]
PARENT = TOOLS.parent/'attention-owned-pyannote-amd'
EXACT = ['protocol.py', 'candidate_protocol.py', 'qualify_outputs.py', 'identity_probes.py',
         'checks.py', 'remote.py', 'test_exact.py', 'test_semantics.py', 'test_prerequisites.py']


def verify_scope():
    closed = ROOT/'artifacts/parakeet-attention-owned-pyannote-amd-20260928/closed.json'
    assert pin(closed)['sha256'] == 'ad6afcc335eb0e810630ae3ead312a6d94a64c0a71cb63ff98918a0c64568503'
    inputs = read(closed)['local_inputs']; files = {}
    for name in [*EXACT, 'test_consumer_reuse.py', 'audit.py', 'remote_prepare.py', 'prepare.py', 'run.py', 'consumer_reuse.py']:
        path = PARENT/name; relative = path.relative_to(ROOT).as_posix()
        assert pin(path) == inputs[relative], name
        files[relative] = pin(path)
    for name in EXACT: assert (TOOLS/name).read_bytes() == (PARENT/name).read_bytes(), name
    assert (TOOLS/'audit.py').read_text() == (PARENT/'audit.py').read_text().replace(
        'Owned attention candidate', 'Collapsed-axis transpose candidate')
    assert (TOOLS/'test_consumer_reuse.py').read_text() == (PARENT/'test_consumer_reuse.py').read_text().replace(
        'test_wrong_census_is_rejected', 'test_wrong_focused_contract_is_rejected').replace(
        "args[2]['actual_model_census']['sha256']", "args[2]['focused_contracts']['sha256']")
    remote = (PARENT/'remote_prepare.py').read_text()
    for before, after in {
        '/dev/shm/lokad-pwt-pyannote-20260927': '/dev/shm/lokad-attention-owned-pyannote-20260928',
        '/dev/shm/lokad-attention-owned-models-20260928': '/dev/shm/lokad-transpose-axis-models-20260928',
        '/dev/shm/lokad-attention-owned-app-20260928': '/dev/shm/lokad-transpose-axis-app-20260928',
        '/dev/shm/lokad-attention-owned-shared-20260928': '/dev/shm/lokad-transpose-axis-shared-20260928',
        'Owned attention candidate': 'Collapsed-axis transpose candidate',
    }.items():
        assert remote.count(before) == 1, before
        remote = remote.replace(before, after)
    assert (TOOLS/'remote_prepare.py').read_text() == remote
    def body(path):
        source = path.read_text()
        return next(ast.get_source_segment(source, node) for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'prepare')
    assert body(TOOLS/'prepare.py') == body(PARENT/'prepare.py')
    return files


if __name__ == '__main__': print(verify_scope())
