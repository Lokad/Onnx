"""Publish the closed Pyannote correctness result; no timing score is assigned."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-pad-current-pyannote-amd-20260926'


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def read(path):return json.loads(path.read_text(encoding='utf8'))


def main():
    proof,value=read(BASE/'closed.json'),read(BASE/'analysis.json')
    assert proof['passed'] and value['passed'] and value['no_performance_measurement']
    assert proof['analysis']==pin(BASE/'analysis.json')
    for name,wanted in proof['files'].items():assert pin(BASE/name)==wanted,name
    assert value['reference_provenance_verified']
    assert value['inventory']['passed'] and value['inventory']['methods']==96 and value['inventory']['unchanged']==95
    assert value['identity_guards']['passed'] and value['identity_guards']['probes']==4
    assert value['identity_guards']['rejection_before_output']
    assert value['consumers']['selected']==value['consumers']['candidate']==value['identity_guards']['consumer']
    assert value['identities']['selected']['Lokad.Onnx.dll']['sha256']=='f3992f40d889a932cd0d15e0323564db801a308f4b24d1848731af30ba3c19f6'
    assert value['identities']['candidate']['Lokad.Onnx.dll']['sha256']=='a74acb17524f23be13e81ade871b2b2ffea2afde5bcdd12e339b75e0197edf10'
    for role in ['selected','candidate']:
        result=value['results'][role]
        assert result['passed'] and (result['arrays'],result['values'],result['public_calls'])==(18,2917107,16)
    result=value['results']['candidate']
    assert result['complete_public_results_exact'] and result['complete_public_semantics_exact']
    assert all(row['bit_identical'] for row in result['comparisons'] if row['reference']=='production')
    maximum=max(row['maximum'] for result in value['results'].values() for row in result['comparisons'] if row['reference']=='native')
    resources=value['resources'];assert len(resources)==6
    samples=sum(row['samples'] for row in resources);peak=max(row['peak_rss'] for row in resources)
    paths=[OUT/'pyannote-20260926.json',OUT/'pyannote-20260926.md']
    assert not any(path.exists() for path in paths)
    with paths[0].open('x',encoding='utf8') as stream:
        json.dump(dict(closure=pin(BASE/'closed.json'),**value),stream,indent=2,allow_nan=False)
    text=f'''# Current padding dispatcher: complete Pyannote correctness

**All numerical and output-ownership checks pass.** Both the qualified root
and padding candidate pass 18 complete arrays / 2,917,107 values and 16 public
calls. Candidate tensors and complete public results, including speaker
centroids, assignments, intervals and status, equal the current root exactly.
The original native scaled-error bound is 1e-4; maximum observed array error
is {maximum:.12g}. Inputs and held outputs remain unchanged across later calls.

One common consumer checks the actual Core and Data hashes against its frozen
arguments. Its real VM build preserves 95 of 96 methods exactly. Main has only
the reviewed argument-count, usage and expected-Data operand changes, including
the resulting branch and exception offsets. This inspector does not record
implementation flags. All four deliberate wrong-Core/wrong-Data probes reject
at the corresponding guard before creating output. Their PID/birth identities
are independently confirmed terminal during collection.

All six jobs and their supervisor finish with code zero. All {samples:,} resource
observations pass; peak owned RSS is {peak:,} bytes. Neither product was rebuilt.
These correctness runs provide no application performance score.

The [Parakeet application result](application-20260926.md) remains 6.023% less
time than current root, or 1.278223 times ORT. All six failed isolated component
repeatability checks remain recorded. Fresh graph and Pyannote application
regressions, followed by actual root/package qualification, precede integration.

[Complete identities, compiled comparison, identity guards, numerical results
and resources]({paths[0].name}).
Closure: `{pin(BASE/'closed.json')['sha256']}`.
Raw evidence: `{BASE.relative_to(ROOT).as_posix()}`.
'''
    with paths[1].open('x',encoding='utf8') as stream:stream.write(text)
    print(json.dumps(dict(passed=True,closure=pin(BASE/'closed.json'),samples=samples,peak_rss=peak,maximum_native_error=maximum)))


if __name__=='__main__':main()
