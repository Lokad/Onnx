"""Publish the complete matched masking diagnosis, including its recovery."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/parakeet-observed-dense-where-profile-resume-amd-20260924'
OUT=Path(__file__).resolve().parent


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def read(path):return json.loads(path.read_text(encoding='utf8'))


def main():
    proof=read(BASE/'closed.json');analysis=read(BASE/'analysis.json')
    assert proof['analysis']==pin(BASE/'analysis.json') and proof['passed']==analysis['passed']
    for key,path in [('collection','capture-collected/capture-collection.json'),('transfer','capture-transfer.json'),
        ('observer_review','capture-collected/observer-review.json')]:assert proof[key]==pin(BASE/path)
    assert proof['initial']==analysis['initial_failure'] and analysis['split_capture']
    assert proof['audit_correction']==pin(BASE/'audit-correction.json')
    correction=read(BASE/'audit-correction.json');assert correction['passed']
    assert analysis['requests']==160 and analysis['observer_reused_exactly'] and analysis['attribution_only']
    spec=read(BASE/'bundle/spec.json');masking=analysis['masking']
    assert masking['kernels']==72 and len(masking['rows'])==72 and masking['shared_ancestors_counted_once']
    compact=dict(closure=pin(BASE/'closed.json'),passed=analysis['passed'],
        current=spec['core'],candidate=spec['candidate_core'],observer=spec['data'],consumer=spec['consumer'],
        requests=analysis['requests'],clips=analysis['clips'],frames=analysis['frames'],
        masking=masking,resources=analysis['resources'],initial_failure=proof['initial'],audit_correction=correction,
        split_capture=True,completed_control_repeated=False,attribution_only=True,
        application_gain_admitted=False,overhead_subtracted=False)
    paths=[OUT/('profile-20260924'+suffix) for suffix in ['.json','.csv','.md']]
    assert not any(p.exists() for p in paths)
    with paths[0].open('x',encoding='utf8') as stream:json.dump(compact,stream,indent=2,allow_nan=False)
    with paths[1].open('x',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['kind','layer','name','calls','selected_seconds','candidate_seconds'])
        writer.writeheader();writer.writerows({k:r[k] for k in writer.fieldnames} for r in masking['rows'])
    verdict='passes' if analysis['passed'] else 'fails'
    text=f'''# Observed Parakeet masking: matched profile

**The prospective masking prediction {verdict}.** All 72 Where kernels total
{masking['selected_seconds']:.6f} s in the current release and
{masking['candidate_seconds']:.6f} s in the candidate: {100*masking['gain']:.2f}% lower.
The required reduction was at least 80%, with improvement in every complete
input group. These clocks include the same observer overhead; they do not
establish complete application performance, and no overhead is subtracted.

| Complete masking family | Current (s) | Candidate (s) | Reduction |
|---|---:|---:|---:|
'''
    for row in masking['families']:
        text+=f"| {row['kind']} | {row['totals']['selected']['seconds']:.6f} | {row['totals']['candidate']['seconds']:.6f} | {100*row['complete_group_gain']:.2f}% |\n"
    groups=masking['complete_groups']
    text+=f'''
The combined group counts shared ancestors once: {groups['selected']['seconds']:.6f} s
to {groups['candidate']['seconds']:.6f} s ({100*masking['complete_group_gain']:.2f}% lower).
All 160 requests across 20 clips and 19 encoded lengths pass the original
public/native checks, with exact current complete results. Every graph and
node interval reconciles; all layers and full group membership are retained.

The original current-release process completed. The candidate was refused
before creation because available memory was below the unchanged 11 GiB
preflight. The complete original code 1, error and collection remain retained.
After sharing verified byte-identical VM artifacts, only the missing candidate
ran in a separate namespace. The joint audit validates each process against
its own supervisor and complete resource/accounting records. No successful
process was repeated, and no check, input, product or limit changed.

The first local audit also caught an obsolete historical decoder packing total.
Current and candidate graphs are identical. The older reference differs only
by the 26,214,400 bytes of recurrent weight preparation already qualified in
the current release. A separate auditor checks that exact difference while
preserving every node and other metadata field; the correction is bound in
the closure. No inference reran and no performance threshold changed.

[All 72 kernel rows](profile-20260924.csv),
[complete groups, identities and recovery proof](profile-20260924.json),
[recovery protocol](../observed-dense-where-profile-resume-amd/README.md).
The original six-process application comparison and release qualifications
remain necessary before changing the product or BENCHMARK.md.

Closure: `{pin(BASE/'closed.json')['sha256']}`.
'''
    with paths[2].open('x',encoding='utf8',newline='\n') as stream:stream.write(text)
    print(json.dumps(dict(passed=analysis['passed'],where_gain=masking['gain'],report=str(paths[2]))))


if __name__=='__main__':main()
