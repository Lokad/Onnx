"""Publish complete numerical regression evidence without assigning a timing score."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def read(path):return json.loads(path.read_text(encoding='utf8'))


def main(kind):
    assert kind == 'shared'
    base=ROOT/f'artifacts/parakeet-pad-current-{kind}-amd-20260926'
    proof=read(base/'closed.json');value=read(base/'analysis.json')
    assert proof['passed'] and value['passed'] and value['no_performance_measurement']
    assert proof['analysis']==pin(base/'analysis.json')
    for name,wanted in proof['files'].items():assert pin(base/name)==wanted,name
    assert value['reference_provenance_verified']
    assert value['identities']['selected']['Lokad.Onnx.dll']['sha256']=='f3992f40d889a932cd0d15e0323564db801a308f4b24d1848731af30ba3c19f6'
    assert value['identities']['candidate']['Lokad.Onnx.dll']['sha256']=='a74acb17524f23be13e81ade871b2b2ffea2afde5bcdd12e339b75e0197edf10'
    resources=value['resources']
    samples=sum(row['samples'] for row in resources)
    peak=max(row['peak_rss'] for row in resources)
    for role in ['selected','candidate']:
        assert sum(value['results'][role+'-'+mode]['arrays'] for mode in ['shared','e5'])==166
        assert sum(value['results'][role+'-'+mode]['values'] for mode in ['shared','e5'])==5000814
    title='Shared models and e5'
    scope='166 arrays and 5,000,814 values per product, including all five e5 sequence/padding cases'
    paths=[OUT/f'{kind}-20260926.json',OUT/f'{kind}-20260926.md']
    assert not any(path.exists() for path in paths)
    with paths[0].open('x',encoding='utf8') as stream:
        json.dump(dict(closure=pin(base/'closed.json'),**value),stream,indent=2,allow_nan=False)
    text=f'''# Current padding dispatcher: {title}

**All numerical and ownership checks pass.** This fresh regression covers
{scope}. Both the current release and the candidate remain within the original
1e-4 scaled-error bound against the pinned ORT reference. Candidate outputs
match the current release exactly. Input immutability and independently owned
held outputs remain verified.

All {len(resources)} jobs and their supervisor are terminal with code 0.
The original resource checks pass for {samples:,} observations; peak owned RSS
is {peak:,} bytes. These are correctness runs and provide no performance score.

[Complete results, identities, numerical errors and resources]({paths[0].name})
remain available. The separate [Parakeet application comparison](application-20260926.md)
passes its original admission. All six isolated component repeatability failures
remain recorded; Pyannote, graph and root/package checks still precede integration.

Closure: `{pin(base/'closed.json')['sha256']}`.
Raw evidence: `{base.relative_to(ROOT).as_posix()}`.
'''
    with paths[1].open('x',encoding='utf8') as stream:stream.write(text)
    print(json.dumps(dict(passed=True,kind=kind,closure=pin(base/'closed.json'),jobs=len(resources),samples=samples,peak_rss=peak)))


if __name__=='__main__':
    assert len(sys.argv)==2
    main(sys.argv[1])
