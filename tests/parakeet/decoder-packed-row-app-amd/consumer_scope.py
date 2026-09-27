"""Keep full requests, exact clocks and original controls with the prospective 1% gate."""
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
PARENT = TOOLS.parent/'direct-depthwise-app-amd'
TRANSPORT = TOOLS.parent/'owned-batch-isolation-parent-app-amd'
PREVIOUS = TOOLS.parent/'rational-sigmoid-app-amd'
NAMES = ['protocol.py', 'remote.py', 'statistics_exact.py', 'audit.py']


def verify_scope():
    for name in NAMES:
        assert (TOOLS/name).read_bytes() == (PARENT/name).read_bytes(), name
    expected = (PARENT/'checks.py').read_text()
    for before, after in [
        ('corpus-at-least-three-percent-gain', 'corpus-at-least-one-percent-gain'),
        ('limit=.97,passed=ratio<=Fraction(97,100)', 'limit=.99,passed=ratio<=Fraction(99,100)'),
        ('Candidate corpus >=3% gain', 'Candidate corpus >=1% gain')]:
        assert expected.count(before) == 1; expected = expected.replace(before, after)
    assert (TOOLS/'checks.py').read_text() == expected
    expected = (PARENT/'test_admission.py').read_text()
    for before, after in [('fixture(candidate=970)', 'fixture(candidate=990)'),
                          ('test_exact_three_percent_boundary', 'test_exact_one_percent_boundary'),
                          ('fixture(971)', 'fixture(991)')]:
        assert expected.count(before) == 1; expected = expected.replace(before, after)
    assert (TOOLS/'test_admission.py').read_text() == expected
    expected = (PREVIOUS/'remote_prepare.py').read_text()
    expected = expected.replace('parakeet-rational-sigmoid-models-20260927',
                                'parakeet-decoder-packed-row-models-20260927')
    expected = expected.replace('current root,rational sigmoid,ORT,ORT,rational sigmoid,current root',
                                'current root,prepared-row candidate,ORT,ORT,prepared-row candidate,current root')
    assert (TOOLS/'remote_prepare.py').read_text() == expected
    marker = "    assert models['consumers']['AudioBenchmark']"
    assert marker+(TOOLS/'prerequisites.py').read_text().split(marker, 1)[1] == marker+(PARENT/'prerequisites.py').read_text().split(marker, 1)[1]
    return True


if __name__ == '__main__': print(verify_scope())
