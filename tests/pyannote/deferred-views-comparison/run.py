"""Run every original managed and native request under the fixed resource bounds."""
from common import *

path = ROOT / 'tests/pyannote/request-comparison/run.py'
namespace = dict(globals(), __name__='original_deferred_views_comparison', __file__=str(Path(__file__)))
exec(compile(path.read_text(encoding='utf8'), str(path), 'exec'), namespace)
if __name__ == '__main__':
    namespace['main']()
