"""Preserve the complete native shared-model and five-input e5 regression lane."""
import sys
from common import *

ORIGINAL = ROOT/'tests/parakeet/reduction-shared/qualify_v2.py'


def load():
    source = ORIGINAL.read_text(encoding='utf8')

    def replace(old, new):
        nonlocal source
        assert source.count(old) == 1, old
        source = source.replace(old, new)

    replace("sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reduction-dispatch'))\nfrom common import ROOT, BASE as MODEL, pin, read, save, verify, terminal, psutil, monitor, rel",
        'from common import ROOT, MODEL, pin, read, save, verify, terminal, psutil, monitor, rel, candidate')
    replace('artifacts/parakeet-reduction-shared-v2-20260921', 'artifacts/pyannote-blocked-spatial-shared-20260922')
    replace('f2c292cb6856e7ec80769e983df1f01512e98ad8d4d6d6ec3d424c6028e8f791', CORE)
    replace("proof = read(MODEL / 'closed.json')\n    assert proof['evidence_passed'] and proof['native_numeric_passed']\n    verify(proof['files'])\n    for identity in proof['terminal_identities']:\n        terminal(identity)", 'proof = candidate()')
    replace("ROOT / 'tests/parakeet/reduction-dispatch/common.py',",
        "ROOT / 'tests/pyannote/blocked-spatial-models/common.py',\n              ROOT / 'tests/parakeet/reduction-shared/qualify_v2.py',\n              ROOT / 'tests/parakeet/portable-models/common.py',")
    replace('limits=dict(preflight_gib=10, rss_gib=8, seconds=900)', 'limits=dict(preflight_gib=12, rss_gib=8, seconds=900)')
    replace("ROOT, [0], 10, 8, 900, False, BASE / 'outputs'", "ROOT, [0], 12, 8, 900, False, BASE / 'outputs'")
    replace("run['preflight']['available'] >= 10 * 1024**3", "run['preflight']['available'] >= 12 * 1024**3")
    namespace = dict(__name__='blocked_original_shared_lane', __file__=str(Path(__file__)))
    exec(compile(source, str(ORIGINAL), 'exec'), namespace)
    return namespace


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ['prepare', 'run', 'audit']
    load()[sys.argv[1]]()
