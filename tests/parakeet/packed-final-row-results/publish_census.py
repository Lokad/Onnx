"""Publish the two actual-model census results and retained preflight refusal."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-packed-final-row-census-resume-amd-20260925'


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def read(path):return json.loads(path.read_text(encoding='utf8'))


def main():
    assert pin(BASE/'closed.json')['sha256']=='fa159300cbe217b8ac08df793031850f2794f2149dc65b89e89b93a5d5692a77'
    closure=read(BASE/'closed.json');analysis=read(BASE/'analysis.json')
    assert closure['passed'] and analysis['passed'] and closure['analysis']==pin(BASE/'analysis.json')
    for name,wanted in closure['files'].items():assert pin(BASE/name)==wanted,name
    assert analysis['original_failure_preserved'] and analysis['completed_mode_not_repeated']
    assert not analysis['release_admitted'] and not analysis['application_scored']
    assert [m['mode'] for m in analysis['modes']]==['512','256']
    for mode in analysis['modes']:
        value=mode['result']
        assert value['passed'] and value['owned_count']==87 and value['retained_maps']==37 and value['initializer_count']==649
        assert value['public_request_passed'] and value['logical_hashes_exact'] and value['identities_preserved']
    initial=analysis['resumed_from']
    assert initial['original_code']==1 and initial['completed_job']=='census-512' and initial['remaining_jobs']==['census-256']
    paths=[OUT/('census-20260925'+suffix) for suffix in ['.json','.md']]
    assert not any(p.exists() for p in paths)
    result=dict(closure=pin(BASE/'closed.json'),analysis=pin(BASE/'analysis.json'),product=analysis['product'],
        modes=[dict(mode=m['mode'],result=m['result'],request=m['request']) for m in analysis['modes']],
        resources=analysis['resources'],initial=initial,terminal_owners=closure['terminal_owners'],
        original_failure_preserved=True,completed_mode_not_repeated=True,application_scored=False,
        release_admitted=False,failed_release_controls=analysis['failed_release_controls'],publisher=pin(Path(__file__)))
    report='''# Parakeet: real-model census for the packed final-row fix

**Both instruction modes pass.** The normal transcriber constructor prepares
exactly 87 owned feed-forward weights while preserving all 37 existing packing
records, their 256 MiB budget and the graph's 649 initializers. Logical weight
values, packed payloads, object identities and shared execution contexts remain
intact before and after the longest public transcription. Its transcript,
tokens, frame decisions and completion result match exactly in both modes.

The tested products are Core49901366/Data01e9e784 from the successful compiled
and focused-contract review. The original compiled census consumer is reused
unchanged. No product or consumer is rebuilt, no forced GC is used, and no
application latency is scored. Peak monitored RSS is 6,380,191,744 bytes.

The first capture completed normal mode, then stopped before launching the
AVX512-disabled worker: available memory was 11,516,035,072 bytes, below the
unchanged 11 GiB requirement. That failed stage and all its evidence remain.
A fresh namespace runs only the unstarted mode, with a bounded wait for the
same threshold. The joint audit verifies both sets of raw results and resource
logs. The completed normal-mode request was not repeated.

This proves the preparation and longest-request contracts. The longest clip
has 225 encoded frames and uses the existing three-row route; full-corpus and
traffic checks are still needed to validate the changed remaining-row path
inside the actual model. No new speedup or release admission is claimed.

[Exact products, public results and retained failure](census-20260925.json),
[focused integration contracts](contracts-20260925.md),
[initial census tools](../packed-final-row-census/README.md),
[recovery tools](../packed-final-row-census-resume/README.md).

Closure: `fa159300cbe217b8ac08df793031850f2794f2149dc65b89e89b93a5d5692a77`.
'''
    with paths[0].open('x',encoding='utf8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    with paths[1].open('x',encoding='utf8') as stream:stream.write(report)
    print(json.dumps(dict(passed=True,modes=2,reports={p.name:pin(p) for p in paths})))


if __name__=='__main__':main()
