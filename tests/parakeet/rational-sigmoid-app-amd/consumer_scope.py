"""Keep the original full-application timing and public-result contracts."""
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
PARENT = TOOLS.parent/'direct-depthwise-app-amd'
TRANSPORT = TOOLS.parent/'owned-batch-isolation-parent-app-amd'
PAD = TOOLS.parent/'pad-current-app-amd'
NAMES = ['protocol.py','remote.py','statistics_exact.py','test_admission.py','checks.py','audit.py']


def verify_scope():
    for name in NAMES:
        assert (TOOLS/name).read_bytes() == (PARENT/name).read_bytes(), name
    original = (PAD/'remote_prepare.py').read_text()
    expected = original.replace('parakeet-pad-current-models-20260926',
                                'parakeet-rational-sigmoid-models-20260927')
    expected = expected.replace('current root,padding candidate,ORT,ORT,padding candidate,current root',
                                'current root,rational sigmoid,ORT,ORT,rational sigmoid,current root')
    assert (TOOLS/'remote_prepare.py').read_text() == expected
    marker = "    assert models['consumers']['AudioBenchmark']"
    tail = marker+(PARENT/'prerequisites.py').read_text().split(marker,1)[1]
    tail = tail.replace("assert len(native['exact_selected_comparisons'])==784 and all(r['bit_identical'] for r in native['exact_selected_comparisons'])",
                        "verify_comparisons(native['selected_comparisons'])")
    assert marker+(TOOLS/'prerequisites.py').read_text().split(marker,1)[1] == tail
    return True


if __name__ == '__main__': print(verify_scope())
