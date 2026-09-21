"""Normalize historical Windows receipt separators before metadata lookup."""
from pathlib import Path

original = Path(__file__).with_name('activation_review.py')
source = original.read_text(encoding='utf8')
old = "            trusted.update(proof['files'])"
new = "            trusted.update({Path(name).as_posix(): wanted for name, wanted in proof['files'].items()})"
assert source.count(old) == 1
source = source.replace(old, new)
old = "        for path in [Path(__file__), BASE / 'analysis.json']:"
new = "        for path in [Path(__file__), ORIGINAL, ROOT / 'artifacts/pyannote-activation-review-failure-20260921/receipt.json', BASE / 'analysis.json']:"
assert source.count(old) == 1
source = source.replace(old, new)
exec(compile(source, str(original), 'exec'), dict(__name__='__main__', __file__=str(Path(__file__)), ORIGINAL=original))
