"""Reuse the existing application lane with one prospective retention gate."""
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
PARENT = TOOLS.parent/'direct-depthwise-app-amd'
NAMES = ['protocol.py', 'remote.py', 'statistics_exact.py', 'audit.py', 'checks.py',
         'test_admission.py', 'run.py', 'remote_prepare.py']


def expected(name):
    text = (PARENT/name).read_text()
    if name == 'checks.py':
        old = "gates.append(dict(name='corpus-at-least-three-percent-gain',candidate_over_current=float(ratio),limit=.97,passed=ratio<=Fraction(97,100)))"
        new = "gates.append(dict(name='corpus-parent-regression-at-most-five-percent',candidate_over_current=float(ratio),limit=1.05,passed=ratio<=Fraction(105,100)))"
        assert text.count(old) == 1
        text = text.replace(old, new).replace('Candidate corpus >=3% gain and no clip >5% slower.',
            'Candidate corpus and every clip <=5% slower than the direct-depthwise parent.')
    elif name == 'test_admission.py':
        text = text.replace('fixture(candidate=970)', 'fixture(candidate=1050)')
        text = text.replace('test_exact_three_percent_boundary', 'test_exact_parent_regression_boundary')
        text = text.replace('fixture(971)', 'fixture(1051)')
    elif name == 'run.py':
        text = text.replace('direct-depthwise-app', 'owned-batch-isolation-parent-app')
        text = text.replace('the direct-depthwise full Parakeet comparison', 'the dispatch-relocation parent comparison')
        text = text.replace('def observe():\n', "def observe():\n    assert not (BASE/'closed.json').exists(), 'Preserve the closed campaign'\n")
    elif name == 'remote_prepare.py':
        text = text.replace('lokad-parakeet-direct-depthwise-models-20260925', 'lokad-parakeet-owned-batch-isolation-models-20260925')
        text = text.replace('M78,direct depthwise,ORT,ORT,direct depthwise,M78', 'depthwise parent,relocation,ORT,ORT,relocation,depthwise parent')
    else:
        assert name in ['protocol.py', 'remote.py', 'statistics_exact.py', 'audit.py']
    return text


def verify_scope():
    for name in NAMES:
        assert (TOOLS/name).read_text() == expected(name), name
    marker = "    assert models['consumers']['AudioBenchmark']"
    before = (PARENT/'prerequisites.py').read_text()
    after = (TOOLS/'prerequisites.py').read_text()
    assert before[before.index(marker):] == after[after.index(marker):]
    return True


if __name__ == '__main__':
    print(verify_scope())
