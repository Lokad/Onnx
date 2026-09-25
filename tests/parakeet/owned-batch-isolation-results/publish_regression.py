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
    assert kind in ['shared','pyannote']
    base=ROOT/f'artifacts/parakeet-owned-batch-isolation-{kind}-amd-20260925'
    proof=read(base/'closed.json');value=read(base/'analysis.json')
    assert proof['passed'] and value['passed'] and value['no_performance_measurement']
    assert proof['analysis']==pin(base/'analysis.json')
    for name,wanted in proof['files'].items():assert pin(base/name)==wanted,name
    assert value['reference_provenance_verified']
    assert value['identities']['selected']['Lokad.Onnx.dll']['sha256']=='f95a13c58354bf07f3b7926b72903c18b1a560a56673297cb9fe001d3541b592'
    assert value['identities']['candidate']['Lokad.Onnx.dll']['sha256']=='e07a45189b348fe55ce76300415c6c35ba6a2bc0d224f1fc13b0b92c303bccba'
    resources=value['resources']
    samples=sum(row['samples'] for row in resources)
    peak=max(row['peak_rss'] for row in resources)
    if kind=='shared':
        for role in ['selected','candidate']:
            assert sum(value['results'][role+'-'+mode]['arrays'] for mode in ['shared','e5'])==166
            assert sum(value['results'][role+'-'+mode]['values'] for mode in ['shared','e5'])==5000814
        title='Shared models and e5'
        scope='166 arrays and 5,000,814 values per product, including all five e5 sequence/padding cases'
    else:
        for role in ['selected','candidate']:
            row=value['results'][role]
            assert row['passed'] and row['arrays']==18 and row['values']==2917107 and row['public_calls']==16
        assert value['results']['candidate']['complete_public_results_exact']
        assert value['consumer_rebuilt'] is False
        assert value['retained_consumer_inventory']==dict(passed=True,methods=96,unchanged=95,literal_only=True,consumer_rebuilt=False,retained_closure='8471ed0c4f9cdca7ba32edf2ba7ef744bedae4c0b0b037af52f1c87642230595')
        title='Pyannote models and public results'
        scope='all segmentation and embedding tensors, complete public results and centroid state'
    paths=[OUT/f'{kind}-20260925.json',OUT/f'{kind}-20260925.md']
    assert not any(path.exists() for path in paths)
    with paths[0].open('x',encoding='utf8') as stream:
        json.dump(dict(closure=pin(base/'closed.json'),**value),stream,indent=2,allow_nan=False)
    text=f'''# Dispatch relocation: {title}

**All numerical and ownership checks pass.** This fresh regression covers
{scope}. Both the current release and the candidate remain within the original
1e-4 scaled-error bound against the pinned ORT reference. Candidate outputs
match the current release exactly. Input immutability and independently owned
held outputs remain verified.

All {len(resources)} jobs and their supervisor are terminal with code 0.
The original resource checks pass for {samples:,} observations; peak owned RSS
is {peak:,} bytes. These are correctness runs and provide no performance score.

[Complete results, identities, numerical errors and resources]({paths[0].name})
remain available. The separate [Parakeet application comparison](release-application-20260925.md)
passes its original admission; remaining release checks still precede integration.

Closure: `{pin(base/'closed.json')['sha256']}`.
Raw evidence: `{base.relative_to(ROOT).as_posix()}`.
'''
    with paths[1].open('x',encoding='utf8') as stream:stream.write(text)
    print(json.dumps(dict(passed=True,kind=kind,closure=pin(base/'closed.json'),jobs=len(resources),samples=samples,peak_rss=peak)))


if __name__=='__main__':
    assert len(sys.argv)==2
    main(sys.argv[1])
