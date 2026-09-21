"""Audit the explicit fixture successor using the unchanged qualification rules."""
from common import *
import audit

TARGET = ROOT / 'artifacts/pyannote-request-contexts-v2-20260921'


if __name__ == '__main__':
    successor = read(TARGET / 'successor-prepared.json'); assert successor['passed']
    assert successor['failure'] == pin(BASE / 'failure-closed.json')
    assert successor['tool'] == pin(TOOLS / 'qualify_v2.py'); verify(successor['source_files'])
    for name, expected in successor['runtime'].items(): assert pin(TARGET / 'runtime' / name) == expected
    audit.BASE = TARGET
    audit.main()
