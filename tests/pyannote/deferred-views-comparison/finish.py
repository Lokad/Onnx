"""Wait for actual application success, then audit and compare once."""
from common import *

path = ROOT / 'tests/pyannote/convolution-portable-comparison/finish.py'
source = path.read_text(encoding='utf8')
changes = [
    ('artifacts/pyannote-convolution-portable-comparison-finish-20260921', 'artifacts/pyannote-deferred-views-comparison-finish-20260922'),
    ('dict(pid=1052836, birth=1790021100.0864675)', 'dict(pid=348468, birth=1790034173.81007)'),
    ('tests/pyannote/convolution-portable-qualification', 'tests/pyannote/deferred-views-applications'),
    ("QUALIFIED / 'qualification.json'", "QUALIFIED / 'processes.json'"),
    ('audit_applications_v2.py', 'audit.py'),
    ("*APPLICATION_TOOLS.glob('*.py')]", "*APPLICATION_TOOLS.glob('*.py'), ROOT / 'tests/pyannote/convolution-portable-comparison/finish.py', ROOT / 'tests/pyannote/convolution-portable-comparison/close_finish.py', ROOT / 'tests/pyannote/sparse-mel-comparison/report.py']")]
for before, after in changes:
    assert source.count(before) >= 1, before
    source = source.replace(before, after)
namespace = dict(globals(), __name__='original_deferred_views_finite_finish', __file__=str(Path(__file__)))
exec(compile(source, str(path), 'exec'), namespace)
if __name__ == '__main__':
    namespace['main']()
