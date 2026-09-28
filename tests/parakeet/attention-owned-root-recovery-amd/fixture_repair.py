"""Replace only test-helper defaults with identical explicit call-site values."""
import hashlib

TEST = 'tests/Lokad.Onnx.Backend.Tests/OwnedAttentionPreparationTests.cs'
ORIGINAL = dict(bytes=12772, sha256='93da19d1fa2eb0c419ecaf90cf2405f09612e4d17e3a4ed12227185345561113')


def fingerprint(data):
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def repair(data):
    assert fingerprint(data) == ORIGINAL
    replacements = [
        (b'Graph(string family = "q", int m = 51, long budget = 0, int n = 1024, int k = 1024)',
         b'Graph(string family, int m, long budget, int n, int k)', 1),
        (b'Graph(family)', b'Graph(family, 51, 0, 1024, 1024)', 1),
        (b'Graph(n: n, k: k)', b'Graph("q", 51, 0, n, k)', 1),
        (b'Graph(budget: 4194304)', b'Graph("q", 51, 4194304, 1024, 1024)', 1),
        (b'Graph(m: m)', b'Graph("q", m, 0, 1024, 1024)', 1),
        (b'Graph(m: 1)', b'Graph("q", 1, 0, 1024, 1024)', 1),
        (b'Graph()', b'Graph("q", 51, 0, 1024, 1024)', 5),
    ]
    for old, new, count in replacements:
        assert data.count(old) == count, (old, data.count(old))
        data = data.replace(old, new)
    return data


def verify_source_map(original, recovered, fixture):
    assert len(original) == len(recovered) == 446
    assert original[TEST] == ORIGINAL
    assert set(original) == set(recovered)
    assert {n for n in original if original[n] != recovered[n]} == {TEST}
    assert recovered[TEST] == fingerprint(fixture)
    return True
