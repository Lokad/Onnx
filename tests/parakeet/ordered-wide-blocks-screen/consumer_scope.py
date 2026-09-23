"""Retain the compiled-consumer source, worker and collection checks without new timing code."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'wide-entry-first-use-screen'


def verify_scope():
    for name in ['Screen.cs', 'Prototype.csproj', 'fixtures.py', 'remote.py', 'protocol.py', 'run.py', 'audit.py']:
        if name in ['Screen.cs', 'Prototype.csproj']:
            assert (HERE / name).read_bytes() == (PARENT / name).read_bytes(), name
        expected = (PARENT / name).read_text()
        if name == 'protocol.py':
            expected = expected.replace("['sdk-version','consumer-restore','consumer-build','current-screen0-512'", "['current-screen0-512'")
        if name == 'run.py':
            expected = expected.replace('wide-entry-first-use', 'ordered-wide-blocks')
            expected = expected.replace('actual-DLL M50 numerical qualification', 'actual-DLL M55 complete-call comparison')
            expected = expected.replace("mode='w|gz') as tar:", "mode='w|gz',dereference=True) as tar:")
        if name == 'audit.py':
            expected = expected.replace("limit=LIMITS['build_preflight_available' if row['name'] in JOBS[:3] else 'preflight_available']", "limit=LIMITS['preflight_available']")
            expected = expected.replace("    assert (folder/'logs/sdk-version.stdout').read_text().strip().endswith('10.0.204')", "    assert built['reused_consumer'] and built['consumer']==payload['consumer']")
            expected = expected.replace("for run in state['runs'][3:]:", "for run in state['runs']:")
        assert (HERE / name).read_text() == expected, name
    return True


if __name__ == '__main__':
    print(verify_scope())
