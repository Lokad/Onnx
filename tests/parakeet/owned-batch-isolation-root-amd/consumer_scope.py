"""Reuse root/package validation, adding the seven public depthwise cases."""
from pathlib import Path
import re
from protocol import pin
from new_cases import NEW_CASES, SKIPPED_CASES, OWNED_CASES, FRAMES, FACTS, DEPTHWISE_FACTS, DEPTHWISE_CASES
from source_scope import verify_templates, TEMPLATES, TEST, DEPTHWISE, SOURCE

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[2]
ORIGINAL = TOOLS.parent/'packed-final-row-root-amd'
GRAPH_TOOLS = TOOLS.parent/'owned-batch-isolation-pyannote-app-amd'


def verify_scope():
    files = {}
    for name in ['protocol.py', 'remote.py', 'admission.py', 'audit.py', 'warning_census.py', 'test_guards.py']:
        source = ORIGINAL/name
        assert (TOOLS/name).read_bytes() == source.read_bytes(), name
        files[source.relative_to(ROOT).as_posix()] = pin(source)
    for name in ['checks.py', 'test_census.py']:
        source = ORIGINAL/name
        expected = source.read_text(encoding='utf8').replace('3277', '3281').replace('3539', '3546').replace('3449', '3456')
        assert (TOOLS/name).read_text(encoding='utf8') == expected, name
        files[source.relative_to(ROOT).as_posix()] = pin(source)
    source = GRAPH_TOOLS/'graph_prerequisite.py'
    assert (TOOLS/source.name).read_bytes() == source.read_bytes()
    files[source.relative_to(ROOT).as_posix()] = pin(source)
    owned, depthwise = verify_templates()
    assert OWNED_CASES == owned['new_backend_cases']['passed']
    assert SKIPPED_CASES['backend'] == owned['new_backend_cases']['skipped']
    assert NEW_CASES['backend'] == sorted(OWNED_CASES+SKIPPED_CASES['backend']+DEPTHWISE_CASES)
    assert len(DEPTHWISE_FACTS) == len(set(DEPTHWISE_FACTS)) == depthwise['expected_public_facts'] == 7
    text = TEMPLATES[DEPTHWISE].read_text(encoding='utf8')
    assert re.findall(r'\[Fact\]\s+public void (\w+)\(', text) == DEPTHWISE_FACTS
    assert len(depthwise['geometries']) == 59
    name = 'tests/Lokad.Onnx.Tensors.Tests/SliceDenseConversionTests.cs'
    qualified = ROOT/'artifacts/parakeet-slice-dense-conversion-tests-amd-20260924/bundle/source'/name
    assert (SOURCE/'source'/name).read_bytes() == qualified.read_bytes()
    text = qualified.read_text(encoding='utf8')
    assert list(map(int, re.search(r'Frames => new\[\]\s*\{([^}]+)\}', text).group(1).split(','))) == FRAMES
    assert sorted(re.findall(r'\[Fact\]\s+public void (\w+)\(', text)) == sorted(FACTS)
    assert text.count('[Theory]') == 1 and len(NEW_CASES['tensors']) == 26 and not SKIPPED_CASES['tensors']
    for source in [*TEMPLATES.values(), qualified, ORIGINAL/'public_tests.py', TOOLS.parent/'direct-depthwise-public-tests/prepare.py']:
        files[source.relative_to(ROOT).as_posix()] = pin(source)
    source = ORIGINAL/'run.py'
    expected = source.read_text(encoding='utf8').replace('packed-final-row-root', 'owned-batch-isolation-root')
    expected = expected.replace('def observe():\n', "def observe():\n    assert not (BASE/'closed.json').exists(), 'Preserve the closed campaign'\n")
    assert (TOOLS/'run.py').read_text(encoding='utf8') == expected
    files[source.relative_to(ROOT).as_posix()] = pin(source)
    return files


if __name__ == '__main__':
    print(verify_scope())
