"""Publish the complete compact attribution from its immutable evidence."""
from pathlib import Path
import json
from analyze import CAPTURE, OUT, ROOT, pin, read

assert pin(CAPTURE/'closed.json')['sha256'] == '473104209f3a9161dc0600df123581fbb6fe626bea8894be69388b953277332e'
assert pin(OUT/'closed.json')['sha256'] == '75f3b25985a8ff4efdc7465de0444b3e667deb0c5922755b0da8a8a5770601c4'
for folder in [CAPTURE, OUT]:
    proof = read(folder/'closed.json'); assert proof['passed'] and proof['diagnostic_only']
    for name, wanted in proof['files'].items(): assert pin(folder/name) == wanted, name
target = ROOT/'tests/parakeet/decoder-lstm-layout-profile-results/pointwise-tail-runtime-observations-20260927.json'
with target.open('x', encoding='utf8') as stream:
    json.dump(dict(**read(OUT/'analysis.json'), analysis_closure=pin(OUT/'closed.json')),
        stream, indent=2, allow_nan=False); stream.write('\n')
print(json.dumps(dict(passed=True, published=pin(target))))
