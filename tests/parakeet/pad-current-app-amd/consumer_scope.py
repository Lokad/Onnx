"""Preserve the original complete-transcription protocol and three-percent gate."""
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
PARENT = TOOLS.parent/'direct-depthwise-app-amd'
TRANSPORT = TOOLS.parent/'owned-batch-isolation-parent-app-amd'
NAMES = ['protocol.py','remote.py','statistics_exact.py','test_admission.py','checks.py','audit.py']


def verify_scope():
    for name in NAMES:
        assert (TOOLS/name).read_bytes() == (PARENT/name).read_bytes(), name
    expected = (TRANSPORT/'remote_prepare.py').read_text()
    expected = expected.replace('parakeet-owned-batch-isolation-models-20260925',
                                'parakeet-pad-current-models-20260926')
    expected = expected.replace("failed_graph_cases=stage['failed_graph_cases']",
                                "failed_component_controls=stage['failed_component_controls']")
    expected = expected.replace('depthwise parent,relocation,ORT,ORT,relocation,depthwise parent',
                                'current root,padding candidate,ORT,ORT,padding candidate,current root')
    assert (TOOLS/'remote_prepare.py').read_text() == expected
    marker = "    assert models['consumers']['AudioBenchmark']"
    original_tail = marker+(PARENT/'prerequisites.py').read_text().split(marker,1)[1]
    actual_tail = marker+(TOOLS/'prerequisites.py').read_text().split(marker,1)[1]
    assert actual_tail == original_tail
    return True


if __name__ == '__main__':
    print(verify_scope())
