"""Correct the exported schema URI check; retain every stack-accounting gate."""
from pathlib import Path

path = Path(__file__).with_name('stacks.py')
source = path.read_text(encoding='utf8')
old = "document['$schema'].endswith('speedscope/file-format-schema.json')"
assert source.count(old) == 1
source = source.replace(old, "document['$schema'] == 'https://www.speedscope.app/file-format-schema.json'")
exec(compile(source, str(path), 'exec'))
