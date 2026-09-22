"""Use the unchanged complete shared-model lane with the composed Core identity."""
import sys
from common import *

ORIGINAL = ROOT / 'tests/parakeet/reduction-shared/qualify_v2.py'


def load():
    source = ORIGINAL.read_text(encoding='utf8')
    def replace(old, new):
        nonlocal source
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    replace("sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reduction-dispatch'))\nfrom common import ROOT, BASE as MODEL, pin, read, save, verify, terminal, psutil, monitor, rel",
            'from common import ROOT, MODEL, pin, read, save, verify, terminal, psutil, monitor, rel, candidate')
    replace('artifacts/parakeet-reduction-shared-v2-20260921', 'artifacts/parakeet-single-panel-shared-20260922')
    replace('f2c292cb6856e7ec80769e983df1f01512e98ad8d4d6d6ec3d424c6028e8f791', CORE)
    replace("proof = read(MODEL / 'closed.json')\n    assert proof['evidence_passed'] and proof['native_numeric_passed']\n    verify(proof['files'])\n    for identity in proof['terminal_identities']:\n        terminal(identity)", 'proof = candidate()')
    replace("ROOT / 'tests/parakeet/reduction-dispatch/common.py',",
            "ROOT / 'tests/parakeet/single-panel-models/common.py',\n              ROOT / 'tests/parakeet/reduction-shared/qualify_v2.py',\n              MODEL / 'external-operand-proof.json',")
    namespace = dict(__name__='original_shared_lane', __file__=str(Path(__file__)))
    exec(compile(source, str(ORIGINAL), 'exec'), namespace)
    return namespace


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'run', 'audit')
    load()[sys.argv[1]]()
