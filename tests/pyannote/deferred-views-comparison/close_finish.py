"""Independently close the finite continuation after its actual exit."""
from common import *

path = ROOT / 'tests/pyannote/convolution-portable-comparison/close_finish.py'
source = path.read_text(encoding='utf8')
before = 'artifacts/pyannote-convolution-portable-comparison-finish-20260921'
assert source.count(before) == 1
source = source.replace(before, 'artifacts/pyannote-deferred-views-comparison-finish-20260922')
namespace = dict(globals(), __name__='original_deferred_views_finish_audit', __file__=str(Path(__file__)))
exec(compile(source, str(path), 'exec'), namespace)
if __name__ == '__main__':
    namespace['main']()
