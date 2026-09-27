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
    for name in ['candidate_protocol.py', 'qualify_outputs.py', 'identity_probes.py', 'test_semantics.py']:
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
    return files


if __name__ == '__main__': print(verify_scope())
