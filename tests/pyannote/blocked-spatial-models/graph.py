"""Reuse all established native/public assertions with the corrected product."""
import sys
from common import *

ORIGINAL = ROOT/'tests/pyannote/single-panel-models/graph.py'


def load():
    source = ORIGINAL.read_text(encoding='utf8')

    def replace(old, new):
        nonlocal source
        assert source.count(old) == 1, old
        source = source.replace(old, new)

    replace("PRIOR = ROOT / 'artifacts/parakeet-portable-models-20260922/pyannote'",
            "PRIOR = ROOT / 'artifacts/pyannote-single-panel-models-20260922'")
    replace("for p in [MODEL / 'closed.json', OLD / 'closed.json', PRIOR / 'closed.json', PRIOR / 'analysis.json',",
            "for p in [ROOT / 'tests/pyannote/single-panel-models/graph.py', ROOT / 'tests/pyannote/blocked-spatial-models/common.py',\n"
            "              MODEL / 'closed.json', ROOT / 'artifacts/pyannote-blocked-spatial-package-20260922/closed.json', OLD / 'closed.json', PRIOR / 'closed.json', PRIOR / 'analysis.json',")
    replace('ROOT, [0], 10, 8, 900, False, BASE / \'output\'', 'ROOT, [0], 12, 8, 900, False, BASE / \'output\'')
    replace("{'pyannote': (10, 8, 900, True)}", "{'pyannote': (12, 8, 900, True)}")
    namespace = dict(__name__='blocked_original_graph_lane', __file__=str(Path(__file__)))
    exec(compile(source, str(ORIGINAL), 'exec'), namespace)
    return namespace


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ['prepare', 'run', 'audit']
    load()[sys.argv[1]]()
