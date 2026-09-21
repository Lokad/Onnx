"""Correct synthetic graph declarations while retaining every candidate byte."""
from pathlib import Path

path = Path(__file__).with_name('prepare_v3.py')
source = path.read_text(encoding='utf8')
changes = [
    ("failure = SECOND / 'failure-closed.json'", "failure = common.ROOT / 'artifacts/pyannote-convolution-pool-v3-20260921/failure-closed.json'"),
    ('1124373d12b1ee8ce8f30eb1ff82d8a8d1fd197da3ef2d2ea72105b55d0e4ca1', '4930f03bfefa918577b20f4d2d8fcaec7b0248717ede0a5e4770405f4c753ce8'),
    ("common.BASE = common.ROOT / 'artifacts/pyannote-convolution-pool-v3-20260921'", "common.BASE = common.ROOT / 'artifacts/pyannote-convolution-pool-v4-20260921'"),
    ("namespace = dict(__name__='proven_initializer_rename'", "replace(\"TOOLS / 'ConvolutionPoolTests.cs'\", \"TOOLS / 'ConvolutionPoolTestsV2.cs'\")\nnamespace = dict(__name__='corrected_synthetic_graph'")]
for old, new in changes:
    assert source.count(old) == 1, old
    source = source.replace(old, new)
exec(compile(source, str(path), 'exec'), dict(__name__='corrected_fixture_preparation', __file__=str(path)))
