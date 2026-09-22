"""Preserve the completed caller checks followed by the erroneous case count."""
import acceptance

c = acceptance.configured()
source = (c['TOOLS'] / 'close_failure.py').read_text(encoding='utf8')
source = source.replace('from prepare import *', '')
old = 'Caller bits rows=32 reduction=64 block=1 groups=2 bias=True pattern=special policy=Auto pass=0 batch=1 index=214 old=ffc00000 new=7fc12345'
assert source.count(old) == 1
source = source.replace(old, 'System.IO.InvalidDataException: Complete schedule')
exec(compile(source, str(c['TOOLS'] / 'close_failure.py'), 'exec'), c)
