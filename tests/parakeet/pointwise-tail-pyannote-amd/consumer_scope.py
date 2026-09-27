"""Preserve original Pyannote workers and checks; adapt only paths and provenance."""
import ast
from pathlib import Path
from protocol import pin, read

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[2]
PARENT = TOOLS.parent/'decoder-lstm-layout-pyannote-amd'
EXACT = ['protocol.py', 'candidate_protocol.py', 'qualify_outputs.py', 'identity_probes.py',
         'checks.py', 'remote.py', 'test_exact.py', 'test_semantics.py',
         'test_prerequisites.py', 'test_consumer_reuse.py']


def remote_preparation():
    source = (PARENT/'remote_prepare.py').read_text()
    replacements = {
        '/dev/shm/lokad-parakeet-decoder-packed-row-pyannote-20260927': '/dev/shm/lokad-lstmlayout-pyannote-20260927',
        '/dev/shm/lokad-lstmlayout-models-20260927': '/dev/shm/lokad-pwt-models-20260927',
        '/dev/shm/lokad-lstmlayout-app-20260927': '/dev/shm/lokad-pwt-app-20260927',
        '/dev/shm/lokad-lstmlayout-shared-20260927': '/dev/shm/lokad-pwt-shared-20260927',
        'Prepared LSTM layout candidate': 'Pointwise remainder candidate'}
    for before, after in replacements.items():
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    assert source.count('copy_function=os.link') == 3
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
    inputs = read(ROOT/'artifacts/parakeet-decoder-lstm-layout-pyannote-amd-20260927/closed.json')['local_inputs']
    files = {}
    for name in [*EXACT, 'audit.py', 'remote_prepare.py', 'prepare.py', 'run.py', 'consumer_reuse.py']:
        path = PARENT/name; relative = path.relative_to(ROOT).as_posix()
        assert pin(path) == inputs[relative], name
        files[relative] = pin(path)
    for name in EXACT: assert (TOOLS/name).read_bytes() == (PARENT/name).read_bytes(), name
    assert (TOOLS/'audit.py').read_text() == (PARENT/'audit.py').read_text().replace(
        'Prepared LSTM layout candidate', 'Pointwise remainder candidate')
    assert (TOOLS/'remote_prepare.py').read_text() == remote_preparation()
    def body(path):
        source = path.read_text()
        return next(ast.get_source_segment(source, node) for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'prepare')
    assert body(TOOLS/'prepare.py') == body(PARENT/'prepare.py')
    return files


if __name__ == '__main__': print(verify_scope())
