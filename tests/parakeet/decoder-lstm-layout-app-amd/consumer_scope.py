"""Preserve all original requests, exact clocks, controls and admission gates."""
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
PARENT = TOOLS.parent/'decoder-packed-row-app-amd'
TRANSPORT = TOOLS.parent/'owned-batch-isolation-parent-app-amd'
NAMES = ['protocol.py', 'remote.py', 'statistics_exact.py', 'audit.py', 'checks.py', 'test_admission.py']


def verify_scope():
    for name in NAMES: assert (TOOLS/name).read_bytes() == (PARENT/name).read_bytes(), name
    expected = (PARENT/'remote_prepare.py').read_text()
    for before, after in [
        ('parakeet-decoder-packed-row-models-20260927', 'lstmlayout-models-20260927'),
        ('current root,prepared-row candidate,ORT,ORT,prepared-row candidate,current root',
         'current root,LSTM layout candidate,ORT,ORT,LSTM layout candidate,current root'),
        ("        for name,wanted in read(folder/'payload.json')['files'].items():assert pin(folder/name)==wanted,name",
         "        for name,wanted in read(folder/'payload.json')['files'].items():\n            if label == 'models' or name.startswith(('assets/', 'runtime/')): assert pin(folder/name)==wanted,name")]:
        assert expected.count(before) == 1; expected = expected.replace(before, after)
    assert (TOOLS/'remote_prepare.py').read_text() == expected
    marker = "    assert models['consumers']['AudioBenchmark']"
    assert marker+(TOOLS/'prerequisites.py').read_text().split(marker, 1)[1] == marker+(PARENT/'prerequisites.py').read_text().split(marker, 1)[1]
    return True


if __name__ == '__main__': print(verify_scope())
