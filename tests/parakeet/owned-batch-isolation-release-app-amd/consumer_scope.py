"""Reuse release-comparison execution, numerical checks and all score gates."""
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
PARENT = TOOLS.parent/'direct-depthwise-app-amd'
RELEASE = TOOLS.parent/'packed-final-row-release-app-amd'
NAMES = ['protocol.py', 'remote.py', 'statistics_exact.py', 'checks.py', 'test_admission.py', 'audit.py', 'run.py']


def expected(name):
    source = (RELEASE if name == 'audit.py' else PARENT)/name
    text = source.read_text()
    if name == 'run.py':
        text = text.replace('direct-depthwise-app', 'owned-batch-isolation-release-app')
        text = text.replace('the direct-depthwise full Parakeet comparison', 'the dispatch-relocation release comparison')
        text = text.replace('def observe():\n', "def observe():\n    assert not (BASE/'closed.json').exists(), 'Preserve the closed campaign'\n")
    return source, text


def verify_scope():
    for name in NAMES:
        source, text = expected(name)
        assert (TOOLS/name).read_text() == text, name
    return True


if __name__ == '__main__':
    print(verify_scope())
