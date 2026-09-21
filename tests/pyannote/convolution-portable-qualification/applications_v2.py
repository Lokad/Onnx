"""Port the already-qualified CLI pipe draining fix into a new suite copy."""
import difflib
from common import *

FAILED = ROOT / 'artifacts/pyannote-convolution-pool-applications-20260921'
BASE = ROOT / 'artifacts/pyannote-convolution-portable-applications-20260921'
ORIGINAL_WRAPPER = TOOLS / 'applications.py'


def correct_cli(source):
    relative = 'tests/Lokad.Onnx.Backend.Tests/CliExitCodeTests.cs'
    path = source / relative
    before = path.read_text(encoding='utf8')
    assert before.count('        if (!p.WaitForExit(180000))') == 1 and before.count('        return p.ExitCode;') == 1
    after = before.replace('        if (!p.WaitForExit(180000))',
        '        // Drain both pipes while the CLI runs; verbose output can exceed their\n'
        '        // capacity before the process reaches its exit-code assertion.\n'
        '        var stdout = p.StandardOutput.ReadToEndAsync();\n'
        '        var stderr = p.StandardError.ReadToEndAsync();\n'
        '        if (!p.WaitForExit(180000))').replace('        return p.ExitCode;',
        '        Task.WhenAll(stdout, stderr).GetAwaiter().GetResult();\n        return p.ExitCode;')
    assert after == (ROOT / relative).read_text(encoding='utf8')
    path.write_text(after, encoding='utf8')
    save(BASE / 'cli-test-correction.json', dict(path=relative, original=pin(MODEL / 'source' / relative),
        corrected=pin(path), existing_root_correction=pin(ROOT / relative),
        diff=''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True))),
        scope='Only drain redirected test pipes; same CLI, timeout and exit-code assertions.'))


def load():
    source = ORIGINAL_WRAPPER.read_text(encoding='utf8')
    anchor = "    namespace = dict(globals(), __name__='original_application_lane', __file__=str(Path(__file__)))"
    assert source.count(anchor) == 1
    source = source.replace(anchor,
        '    replace("    runtime = MODEL / \'runtime\'", "    correct_cli(source)\\n    runtime = MODEL / \'runtime\'")\n' + anchor)
    namespace = dict(globals(), __name__='corrected_application_wrapper', __file__=str(Path(__file__)))
    exec(compile(source, str(ORIGINAL_WRAPPER), 'exec'), namespace)
    original_prerequisites = namespace['prerequisites']
    def prerequisites():
        path = FAILED / 'failure-closed.json'
        assert pin(path)['sha256'] == '5ea6627baa95212378a2f8dab65ce7f7b66a94afbea6f6b72243d484c9c95683'
        failure = read(path)
        assert not failure['passed'] and not failure['public_inference_started']
        verify(failure['files'])
        for identity in failure['identities']:
            terminal(identity)
        prepared = original_prerequisites()
        prepared['files'][rel(path)] = pin(path)
        for path in [ROOT / 'tests/Lokad.Onnx.Backend.Tests/CliExitCodeTests.cs', ORIGINAL_WRAPPER,
                     Path(__file__), TOOLS / 'audit_applications_v2.py']:
            prepared['files'][rel(path)] = pin(path)
        return prepared
    namespace['prerequisites'] = prerequisites
    return namespace['load']()


if __name__ == '__main__':
    load()['main']()
