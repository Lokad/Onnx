"""Run the original full application auditor without changing its assertions."""
from prepare import PARENT
source = PARENT/'audit.py'
exec(compile(source.read_text(),str(source),'exec'))
