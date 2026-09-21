"""Retain all Parakeet outputs and the same three known native discrepancies."""
import sys
from common import *

path = ROOT / 'tests/pyannote/convolution-portable-qualification/parakeet.py'
source = path.read_text(encoding='utf8')
changes = [
    ('artifacts/pyannote-convolution-portable-parakeet-20260921', 'artifacts/pyannote-deferred-views-parakeet-20260922'),
    ("AUDITOR, MONITOR, Path(__file__), TOOLS / 'common.py'",
     "AUDITOR, MONITOR, Path(__file__), TOOLS / 'common.py', ROOT / 'tests/pyannote/convolution-portable-qualification/parakeet.py', ROOT / 'tests/pyannote/convolution-portable-qualification/common.py'")]
for before, after in changes:
    assert source.count(before) == 1, before
    source = source.replace(before, after)
namespace = dict(globals(), __name__='original_deferred_views_parakeet', __file__=str(Path(__file__)))
exec(compile(source, str(path), 'exec'), namespace)
if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'run', 'audit')
    namespace[sys.argv[1]]()
