"""Use every original graph/ownership/counter check on the new exact runtime."""
import shutil
import sys
import traceback
from common import *

original = ROOT / 'tests/pyannote/convolution-portable-rows/probe.py'
source = original.read_text(encoding='utf8')
changes = [
    ('from phase_audit import audit_preparation', ''),
    ("prepared=read(BASE/'prepared.json');preparation=audit_preparation(BASE,sys.modules['common'])",
     "prepared=read(BASE/'prepared.json'); closure=read(BASE/'focused-closed.json'); assert closure['passed']; verify(closure['files']); preparation=read(BASE/'focused-analysis.json'); [terminal(i) for i in closure['identities']]"),
    ("original/'audit.py',ROOT/'tests/pyannote/convolution-pool/phase_audit.py',*TOOLS.iterdir()",
     "original/'audit.py',ROOT/'tests/pyannote/convolution-portable-rows/probe.py',BASE/'focused-closed.json',*TOOLS.iterdir()")]
for before, after in changes:
    assert source.count(before) == 1, before
    source = source.replace(before, after)
namespace = dict(globals(), __name__='original_deferred_views_probe', __file__=str(original))
exec(compile(source, str(original), 'exec'), namespace)

if __name__ == '__main__':
    namespace['main']()
