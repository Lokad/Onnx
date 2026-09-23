"""Require the complete fixed warmed workload and all its acceptance checks."""
from pathlib import Path
from protocol import pin

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[2]
ORIGINAL = ROOT/'tests/benchmarks/warmed-release-amd-v2'


def verify_scope():
    files = {}
    for name in ['checks.py', 'statistics.py', 'native.py', 'test_statistics.py',
                 'test_inventory.py', 'remote.py', 'protocol.py', 'audit.py']:
        source = ORIGINAL/name
        expected = source.read_bytes()
        if name == 'protocol.py':
            before = b"BUILD_JOBS=['sdk-version','consumer-restore','consumer-build','bridge-restore','bridge-build','consumer-inventory']"
            assert expected.count(before) == 1
            expected = expected.replace(before, b'BUILD_JOBS=[]')
        if name == 'audit.py':
            expected = expected.replace(b"c/'consumer-inventory/instructions.json'", b"c/'evidence/warmed-consumer/instructions.json'")
            expected = expected.replace(b"c/'consumer-inventory/review.json'", b"c/'evidence/warmed-consumer/review.json'")
            before = b"    assert (c/'logs/sdk-version.stdout').read_text().strip().endswith('10.0.204')"
            assert expected.count(before) == 1
            expected = expected.replace(before, b"    assert built['reused_consumer'] and built['consumer'] == read(c/'stage.json')['consumer']\n    assert consumer['consumer'] == built['consumer']")
        assert (TOOLS/name).read_bytes() == expected, name
        files[source.relative_to(ROOT).as_posix()] = pin(source)
    return files


if __name__ == '__main__':
    print(verify_scope())
