"""Close the second actual-caller failure without changing either artifact tree."""
import successor

c=successor.configured()
source=(c['TOOLS'] / 'close_failure.py').read_text(encoding='utf8')
assert source.count('from prepare import *')==1
source=source.replace('from prepare import *','')
old='Caller bits rows=32 reduction=64 block=1 groups=2 bias=True pattern=special policy=Auto pass=0 batch=1 index=214 old=ffc00000 new=7fc12345'
new='Caller bits rows=32 reduction=64 block=7 groups=2 bias=True pattern=special policy=Auto pass=0 batch=1 index=1720 old=7fc12345 new=ffc00000'
assert source.count(old)==1;source=source.replace(old,new)
exec(compile(source,str(c['TOOLS'] / 'close_failure.py'),'exec'),c)
