"""Execute the unchanged six-worker public/native comparison with frozen inputs."""
from common import *

path = ROOT / 'tests/pyannote/request-comparison/run.py'
namespace = dict(globals(), __name__='original_comparison_runner', __file__=str(Path(__file__)))
exec(compile(path.read_text(encoding='utf8'), str(path), 'exec'), namespace)
if __name__ == '__main__':
    namespace['main']()
