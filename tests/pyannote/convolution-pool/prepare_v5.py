"""Require safe recovery and restored reuse after the failed run drains the cache."""
from pathlib import Path
path = Path(__file__).with_name('prepare_v4.py')
source = path.read_text(encoding='utf8')
changes = [
    ("failure = common.ROOT / 'artifacts/pyannote-convolution-pool-v3-20260921/failure-closed.json'", "failure = common.ROOT / 'artifacts/pyannote-convolution-pool-v4-20260921/failure-closed.json'"),
    ('4930f03bfefa918577b20f4d2d8fcaec7b0248717ede0a5e4770405f4c753ce8', 'a5ce61a0f9a4077f92142e8659394a9ae009a1924ffc9be2b09a47a35f65a7f7'),
    ("common.BASE = common.ROOT / 'artifacts/pyannote-convolution-pool-v4-20260921'", "common.BASE = common.ROOT / 'artifacts/pyannote-convolution-pool-v5-20260921'"),
    ('ConvolutionPoolTestsV2.cs', 'ConvolutionPoolTestsV3.cs')]
for old, new in changes:
    assert source.count(old) == 1, old
    source = source.replace(old, new)
exec(compile(source, str(path), 'exec'), dict(__name__='recovery_fixture_preparation', __file__=str(path)))
