"""Publish the single packed-row proof without implying model or release admission."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-packed-final-row-proof-amd-20260925'
SOURCE=ROOT/'artifacts/parakeet-packed-final-row-source-20260925'


def pin(path):
    with path.open('rb') as stream:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def read(path):return json.loads(path.read_text(encoding='utf8'))


def main():
    assert pin(BASE/'closed.json')['sha256']=='f4cc1ea804d95014ac878757f8f5107b220a940223b1497bbf6ab55136ba415f'
    closure=read(BASE/'closed.json');assert closure['passed'] and closure['proof_passed']
    for name,wanted in closure['files'].items():assert pin(BASE/name)==wanted,name
    assert closure['analysis']==pin(BASE/'analysis.json')
    value=read(BASE/'analysis.json');assert value['proof_passed'] and value['cases']==98 and not value['failures']
    assert value['no_product_change'] and value['no_model_execution'] and value['no_application_score']
    spec=read(BASE/'bundle/spec.json');rows=[]
    for mode,result in value['modes'].items():
        assert result['passed'] and result['failed']==0
        for case,record in zip(spec['cases'][mode],result['cases'],strict=True):
            assert case['id']==record['id'] and record['passed']
            evidence=record['evidence']
            rows.append(dict(mode=mode,id=case['id'],kind=case['kind'],passed=True,n=case.get('n',''),k=case.get('k',''),m=case.get('m',''),
                             allocated_bytes=evidence.get('allocated_bytes',''),output_sha256=evidence.get('output_sha256','')))
    assert [sum(r['kind']==kind for r in rows) for kind in ['row','route','fallback','unavailable']]==[48,40,9,1]
    assert all(r['allocated_bytes']==0 for r in rows if r['kind']=='row')
    assert pin(SOURCE/'prepared.json')['sha256']=='e3ca64b50ea4a5dd276d90a19779d602f8ed78b6275b9e5448fe3b021bec5e1b'
    candidate=read(SOURCE/'prepared.json')
    assert candidate['proof']==pin(BASE/'closed.json') and candidate['proof_helper']==value['helper']
    assert not candidate['built'] and not candidate['root_product_changed']
    for name,wanted in candidate['source'].items():assert pin(SOURCE/'source'/name)==wanted,name
    paths=[OUT/('proof-20260925'+suffix) for suffix in ['.json','.csv','.md']]
    assert not any(path.exists() for path in paths)
    text='''# Parakeet: packed-input final-row proof

**All 98 correctness cases pass.** One implementation reads the existing packed
weight panels directly while preserving the original reduction order, FMA
operand order and scalar tail. Every result bit matches the original compiled
routine, including special float values, nonzero destinations and partial panels.
The helper allocates zero managed bytes in all 48 checked one-row calls.

| Actual instruction mode | Passed cases |
|---|---:|
| Normal | 47 |
| AVX512 disabled | 47 |
| All hardware intrinsics disabled | 4 |

Forty complete-route cases cover both actual weight shapes, the seven affected
sequence lengths and unchanged even/divisible-by-three routes. Their complete
outputs also match the existing public MatMul. The candidate route receives no
dense B pointer. Guard regions and input bytes remain unchanged. Hardware-disabled
execution checks existing fallbacks and rejects an explicit unsupported intrinsic
request; the new helper is not called there.

Core82c02785 and Data3f80f8cb are unchanged. Only the standalone consumer is built,
with SDK10.0.204/runtime10.0.8, zero warnings and CPU2 ownership. No model or
application benchmark runs in this proof. Peak monitored process RSS is416 MB.

The isolated integration source is now prepared at e3ca64b5:433 files, with the
proved helper copied byte-for-byte, the odd-row reconstruction removed from both
2D and batched routes, and focused graph tests updated. Existing two-/three-row
arithmetic, packing selection, public interfaces and logical fallback remain.
This source has not been built or qualified yet.

Next, inspect compiled scope and require the integrated helper to match this
proved body, run graph/model/public checks, confirm zero actual reconstructions,
and measure a fresh complete application comparison. The original3% gain gate
and prior e5 release failures remain. M77's89/92 timing controls still reject
quantitative attribution; this proof creates no projected application saving.
The qualified repository and BENCHMARK.md remain unchanged.

[Every case](proof-20260925.csv), [identities and evidence](proof-20260925.json),
[helper source](../packed-final-row-probe/PackedFinalRowKernel.cs.txt),
[integration preparer](../packed-final-row-source/prepare.py).

Closure: `f4cc1ea804d95014ac878757f8f5107b220a940223b1497bbf6ab55136ba415f`.
'''
    with paths[0].open('x',encoding='utf8') as stream:
        json.dump(dict(passed=True,proof_only=True,closure=pin(BASE/'closed.json'),analysis=pin(BASE/'analysis.json'),
                       product=value['product'],helper=value['helper'],consumer=value['consumer'],cases=rows,
                       resources=value['resources'],prepared_source=pin(SOURCE/'prepared.json'),source_files=433,
                       model_executed=False,application_scored=False,release_admitted=False,
                       failed_release_controls=value['failed_release_controls'],prior_quantitative_attribution=False,
                       publisher=pin(Path(__file__))),stream,indent=2,allow_nan=False)
    with paths[1].open('x',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with paths[2].open('x',encoding='utf8',newline='\n') as stream:stream.write(text)
    print(json.dumps(dict(published=True,cases=98,proof_only=True,source=pin(SOURCE/'prepared.json'))))


if __name__=='__main__':main()
