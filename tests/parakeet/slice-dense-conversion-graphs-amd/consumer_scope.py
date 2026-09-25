"""Reuse the complete qualified M70 graph protocol, including its e5 warmup."""
from pathlib import Path
from protocol import pin
TOOLS=Path(__file__).resolve().parent;ROOT=TOOLS.parents[2]
ORIGINAL=TOOLS.parent/'observed-dense-where-graphs-amd'
UNCHANGED=['protocol.py','checks.py','checks_e5.py','remote.py','audit.py','native.py','native-e5.py',
 'statistics.py','statistics_base.py','statistics_e5.py','test_statistics.py','test_e5_statistics.py',
 'test_inventory.py','test_dispatch.py']


def verify_scope():
    files={}
    for name in UNCHANGED:
        assert (TOOLS/name).read_bytes()==(ORIGINAL/name).read_bytes(),name
        files[(ORIGINAL/name).relative_to(ROOT).as_posix()]=pin(ORIGINAL/name)
    return files


if __name__=='__main__':print(verify_scope())
