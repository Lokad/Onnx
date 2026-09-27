"""Preserve the original native checks, worker resource policy and identity probes."""
import ast
from pathlib import Path
from protocol import pin, read

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[2]
PARENT = TOOLS.parent/'pad-current-pyannote-amd'


def function(path, name):
    text = path.read_text(encoding='utf8')
    return next(ast.get_source_segment(text, n) for n in ast.parse(text).body
                if isinstance(n, ast.FunctionDef) and n.name == name)


def verify_scope():
    original = read(ROOT/'artifacts/parakeet-pad-current-pyannote-amd-20260926/closed.json')['local_inputs']
    files = {}
    for name in ['protocol.py', 'candidate_protocol.py', 'qualify_outputs.py', 'identity_probes.py',
                 'checks.py', 'remote.py', 'audit.py', 'run.py', 'test_semantics.py']:
        path = PARENT/name; relative = path.relative_to(ROOT).as_posix()
        assert pin(path) == original[relative], name
        files[relative] = pin(path)
    for name in ['candidate_protocol.py', 'qualify_outputs.py', 'identity_probes.py', 'test_semantics.py', 'checks.py']:
        assert (TOOLS/name).read_bytes() == (PARENT/name).read_bytes(), name
    expected = (PARENT/'protocol.py').read_text().replace(
        "JOBS = ['consumer-restore','consumer-build','consumer-inventory','identity-probes','selected','candidate']",
        "JOBS = ['identity-probes','selected','candidate']")
    assert (TOOLS/'protocol.py').read_text() == expected
    for name in ['live', 'size', 'idle']:
        assert function(TOOLS/'remote.py', name) == function(PARENT/'remote.py', name)
    expected = function(PARENT/'remote.py', 'main')
    expected = expected.replace("['logs','tmp','packages','cli-home','http-cache','empty-feed','built']", "['logs','tmp']")
    line, = [s for s in expected.splitlines(True) if s.startswith('    build_env=')]
    expected = expected.replace(line, '').replace('                        job_env=build_env if build else env',
        '                        assert not build\n                        job_env=env')
    assert function(TOOLS/'remote.py', 'main') == expected
    for name in ['semantic', 'semantic_agreement']:
        assert function(TOOLS/'checks.py', name) == function(PARENT/'checks.py', name)
    previous = TOOLS.parent/'decoder-packed-row-pyannote-amd'
    for name in ['protocol.py', 'remote.py', 'checks.py', 'candidate_protocol.py', 'qualify_outputs.py', 'identity_probes.py', 'test_exact.py', 'test_semantics.py']:
        assert (TOOLS/name).read_bytes() == (previous/name).read_bytes(), name
        files[(previous/name).relative_to(ROOT).as_posix()] = pin(previous/name)
    assert (TOOLS/'audit.py').read_text() == (previous/'audit.py').read_text().replace('Prepared single-row candidate', 'Prepared LSTM layout candidate')
    expected = (previous/'remote_prepare.py').read_text().replace('parakeet-rational-sigmoid-pyannote-20260927', 'parakeet-decoder-packed-row-pyannote-20260927')
    for kind in ['models', 'app', 'shared']:
        expected = expected.replace('parakeet-decoder-packed-row-'+kind+'-20260927', 'lstmlayout-'+kind+'-20260927')
    expected = expected.replace('Prepared single-row candidate', 'Prepared LSTM layout candidate')
    expected = expected.replace("        for name, wanted in spec['files'].items(): assert pin(folder/name) == wanted, name",
        "        for name, wanted in spec['files'].items():\n            if label not in ['current', 'previous'] or name.startswith(('assets/', 'graph-reference/', 'runtimes/', 'built/')) or name in ['graph-reference.json', 'evidence/original-manifest.json']:\n                assert pin(folder/name) == wanted, name")
    assert (TOOLS/'remote_prepare.py').read_text() == expected
    for name in ['audit.py', 'remote_prepare.py']:
        files[(previous/name).relative_to(ROOT).as_posix()] = pin(previous/name)
    return files


if __name__ == '__main__': print(verify_scope())
