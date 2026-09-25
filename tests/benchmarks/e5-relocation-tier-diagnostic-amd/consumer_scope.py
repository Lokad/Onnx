"""Limit adaptation to job selection, provenance and the failed-case selector."""
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
PARENT = TOOLS.parent/'e5-direct-tier-diagnostic-v2-amd'


def verify_scope():
    before = (PARENT/'protocol.py').read_text()
    old = "JOBS = ['sdk-version','producer-restore','producer-build','consumer-inventory','observer-inventory','tracer-version']"
    new = "JOBS = ['tracer-version']"
    assert before.count(old) == 1
    assert (TOOLS/'protocol.py').read_text() == before.replace(old, new)
    before = (PARENT/'audit.py').read_text()
    old = "failed=[r for r in read(folder/'evidence/graph-analysis.json')['performance'] if not r['regression_passed']]"
    new = "failed=[r for r in read(folder/'evidence/graph-analysis.json')['performance'] if not r['qualified']]"
    assert before.count(old) == 1
    assert (TOOLS/'audit.py').read_text() == before.replace(old, new)
    before = (PARENT/'run.py').read_text()
    replacements = [
        ("BASE = ROOT/'artifacts/e5-direct-tier-diagnostic-v2-amd-20260925'", "BASE = ROOT/'artifacts/e5-relocation-tier-diagnostic-amd-20260925'"),
        ("REMOTE = '/dev/shm/lokad-e5-direct-tier-diagnostic-v2-20260925'", "REMOTE = '/dev/shm/lokad-e5-relocation-tier-diagnostic-20260925'"),
        ("[*JOBS,'logs','runtimes','export-runtime','reference','evidence','tools','source','previous','previous-observer','bridge']",
         "[*JOBS,'consumer-inventory','observer-inventory','logs','runtimes','export-runtime','reference','evidence','tools','source','previous','previous-observer','bridge']"),
        ("def observe():\n    value =", "def observe():\n    assert not (BASE/'closed.json').exists(), 'Preserve the closed diagnostic'\n    value ="),
    ]
    for old, new in replacements:
        assert before.count(old) == 1
        before = before.replace(old, new)
    assert (TOOLS/'run.py').read_text() == before
    return dict(passed=True, protocol_change='omit five completed build/inventory jobs',
                audit_change='one failed-case selector', consumers_rebuilt=False,
                transport_changes='namespace, collect retained compiled proofs, closed-observation guard',
                call_count=6000, scored=False)


if __name__ == '__main__':
    print(verify_scope())
