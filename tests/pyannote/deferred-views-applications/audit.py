"""All original exact-output, native, ownership and resource gates unchanged."""
from common import *

original = ROOT / 'tests/pyannote/portable-applications/audit.py'
namespace = dict(globals(), __name__='original_deferred_views_application_audit', __file__=str(Path(__file__)))
exec(compile(original.read_text(encoding='utf8'), str(original), 'exec'), namespace)
if __name__ == '__main__':
    namespace['main']()
