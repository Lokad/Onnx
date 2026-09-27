"""Preserve the entire original auditor, changing only the descriptive product label."""
from prepare import PARENT

source = (PARENT/'audit.py').read_text(encoding='utf8')
assert source.count('Current-root padding dispatcher') == 1
source = source.replace('Current-root padding dispatcher', 'Prepared LSTM layout candidate')
if __name__ == '__main__':
    exec(compile(source, str(PARENT/'audit.py'), 'exec'), dict(__name__='__main__'))
