"""Preserve every observation and the original prospective stability/admission gates."""
from common import *

path = ROOT / 'tests/pyannote/request-comparison/audit.py'
namespace = dict(globals(), __name__='original_comparison_auditor', __file__=str(Path(__file__)))
exec(compile(path.read_text(encoding='utf8'), str(path), 'exec'), namespace)
if __name__ == '__main__':
    namespace['main']()
