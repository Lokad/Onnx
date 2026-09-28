"""Reuse every complete-model audit assertion, changing only the candidate label."""
from prepare import LIB, LABELS
source = LIB/'audit.py'
text = source.read_text()
assert text.count('Prepared LSTM grouped weights') == 1
exec(compile(text.replace('Prepared LSTM grouped weights', LABELS['candidate']), str(source), 'exec'))
