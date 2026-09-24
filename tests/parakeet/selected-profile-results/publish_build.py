"""Publish the closed consumer qualification and independently verify its exact source delta."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/parakeet-selected-profile-build-amd-20260924'
PRIOR=ROOT/'artifacts/parakeet-current-profile-build-amd-20260923'
OUT=Path(__file__).resolve().parent


def pin(path):
    with path.open('rb') as stream:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def main():
    paths=['consumer-review-20260924.json','consumer.patch','build-20260924.md']
    assert all(not (OUT/name).exists() for name in paths)
    assert pin(BASE/'closed.json')['sha256']=='b5364b30098c9fa7b375d4d8b404292d7085a315336100e725b451d395cb09b2'
    proof=json.loads((BASE/'closed.json').read_text());assert proof['passed']
    for name,wanted in proof['files'].items():assert pin(BASE/name)==wanted,name
    value=json.loads((BASE/'analysis.json').read_text())
    assert value['inventory']['passed'] and value['inventory']['changed_hash_literals']==2
    before=(PRIOR/'bundle/source/Program.cs').read_text()
    changes={
        '521bae1702849ca23dda586515e7cbabaac2d1eabdff04dc90a7ba76059e93fb':'672e5f303b011e27bb23097a49252c38ddee334938c75895e3c2341df0f3be35',
        'f3b9aa81ee9766797e95216714dec559c5d8b020f8df510cf5f7ee0dda82693a':'065b7a7f28561a37174f9d38ac13ef77bc59c010702e2b64200e9542376eb4c5'}
    for old,new in changes.items():assert before.count(old)==1;before=before.replace(old,new)
    assert before==(BASE/'bundle/source/Program.cs').read_text()
    for name in ['Diagnostic.cs','NpySupport.cs','SampledAudio.csproj']:
        assert (BASE/'bundle/source'/name).read_bytes()==(PRIOR/'bundle/source'/name).read_bytes()
    report=dict(closed=pin(BASE/'closed.json'),analysis_identity=pin(BASE/'analysis.json'),
        inventory_identity=pin(BASE/'collected/inventory/instructions.json'),source_patch=pin(BASE/'bundle/consumer.patch'),
        source_exact_except_two_hashes=True,**value)
    with (OUT/paths[0]).open('x',encoding='utf8') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    with (OUT/paths[1]).open('xb') as stream:stream.write((BASE/'bundle/consumer.patch').read_bytes())
    resources=value['resources'];samples=sum(r['samples'] for r in resources);peak=max(r['peak_rss'] for r in resources)
    text=f'''# Selected-release Parakeet profile consumer

Qualification passes on AMD, SDK 10.0.204 / runtime 10.0.8. Core `672e5f30`
and Data `065b7a7f` are copied unchanged. Consumer `a196f652` is 49,664 bytes.

All 162 consumer methods and their implementation flags are inspected. Of
these, 161 compiled bodies are unchanged; Main contains 890 instructions and
differs only at the two product-hash strings, offsets 374 and 419. Its locals,
branches, stack and exception regions match. Public interfaces match; no method
is added or removed. Both complete-public-call markers retain NoInlining alone.
The source patch changes those same two strings; other source is byte-identical.

Six jobs and all {samples} resource observations pass, peak owned RSS {peak:,} bytes.
Owner 904493 / birth 1790223631.71 and every descendant are terminal. The separate
inventory reader is built from the retained complete-method/flag reader, targeting
SampledAudio and omitting the irrelevant Core/Data special cases. Six adversarial
Python tests reject unintended instructions, method headers, census or flag changes.

[Full qualification summary](consumer-review-20260924.json) and
[exact source patch](consumer.patch) are retained. Complete compiled instructions,
source, process identities and resources remain under
`artifacts/parakeet-selected-profile-build-amd-20260924`.
Closure: `b5364b30098c9fa7b375d4d8b404292d7085a315336100e725b451d395cb09b2`.
Tools frozen at `b8e9e99c`; archive `fd1702ff`, payload `89dedc83`.

This qualifies a separate complete-request diagnostic capture. It establishes
no speed gain, new ORT ratio or product promotion. BENCHMARK.md remains the
qualified selected release.
'''
    with (OUT/paths[2]).open('x',encoding='utf8') as stream:stream.write(text)
    print(json.dumps(dict(passed=True,reports={name:pin(OUT/name) for name in paths})))


if __name__=='__main__':main()
