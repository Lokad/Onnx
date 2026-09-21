"""Apply the unchanged captured-input probe to the corrected fixture qualification."""
from pathlib import Path
path = Path(__file__).with_name('probe.py')
source = path.read_text(encoding='utf8')
assert source.count('pyannote-convolution-pool-v4-20260921') == 1
source = source.replace('pyannote-convolution-pool-v4-20260921', 'pyannote-convolution-pool-v5-20260921')
exec(compile(source, str(path), 'exec'), dict(__name__='__main__', __file__=str(path)))
