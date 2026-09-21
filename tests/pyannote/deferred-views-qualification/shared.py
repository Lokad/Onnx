"""Run and independently audit the unchanged complete shared-model lane."""
import sys
from common import *

path = ROOT / 'tests/pyannote/convolution-portable-qualification/shared.py'
source = path.read_text(encoding='utf8')
changes = [
    ('artifacts/pyannote-convolution-portable-shared-20260921', 'artifacts/pyannote-deferred-views-shared-20260922'),
    ("ROOT / 'tests/pyannote/convolution-portable-qualification/common.py',\\n",
     "ROOT / 'tests/pyannote/deferred-views-qualification/common.py',\\n              ROOT / 'tests/pyannote/convolution-portable-qualification/common.py',\\n              ROOT / 'tests/pyannote/convolution-portable-qualification/shared.py',\\n")]
for before, after in changes:
    assert source.count(before) == 1, before
    source = source.replace(before, after)
namespace = dict(globals(), __name__='original_deferred_views_shared', __file__=str(Path(__file__)))
exec(compile(source, str(path), 'exec'), namespace)
if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'run', 'audit')
    namespace['load']()[sys.argv[1]]()
