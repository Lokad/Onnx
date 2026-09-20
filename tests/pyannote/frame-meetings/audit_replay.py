"""Inspect every new public result and score without native inference replay."""
from pathlib import Path
import sys, math, json
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/pyannote/natural-meetings'))
from common import pin,read,write
from audit import compare
from validation import worker,resources,damaged_worker_checks,damaged_resource_checks
from independent_score import score as independent_score
sys.path.insert(0,str(ROOT/'tests/pyannote/accuracy'))
for name in ['pyannote-dialogue-20260919/metric-python','pyannote-diarization-20260919/python','pyannote-clustering-20260919/python']:
    sys.path.insert(0,str(ROOT/'artifacts'/name))
from diarization_error import score


def main():
    base=ROOT/'artifacts/pyannote-frame-meetings-20260920';collected=base/'collected'
    original=ROOT/'artifacts/pyannote-natural-meetings-20260920'
    receipt=read(collected/'collection.json');assert receipt['terminal']
    for name,wanted in receipt['files'].items():assert pin(collected/name)==wanted,name
    frozen=read(collected/'frozen.json');manifest=read(collected/'manifest.json')
    for name,wanted in frozen['files'].items():assert pin(collected/name)==wanted,name
    for name,wanted in read(collected/'preparation.json')['bindings'].items():assert pin(ROOT/name)==wanted,name
    state=read(collected/'process-managed-run/identity.json');samples=[json.loads(line) for line in (collected/'process-managed-run/samples.jsonl').read_text().splitlines()]
    resources(state,samples,pin(collected/'frozen.json')['sha256'],manifest['limits'],'managed')
    assert read(collected/'process-managed-run/complete.json')['code']==0
    assert not (collected/'process-managed-run/stderr.txt').read_text().strip()
    gaps=[samples[0]['seconds']]+[b['seconds']-a['seconds'] for a,b in zip(samples,samples[1:])]+[state['seconds']-samples[-1]['seconds']]
    assert 0<=min(gaps)<=max(gaps)<10
    actual=read(collected/'process-managed-run/worker/result.json');worker(actual,manifest,frozen,collected,'managed')
    refusal=dict(worker=damaged_worker_checks(actual,manifest,frozen,collected,'managed'),
                 resource=damaged_resource_checks(state,samples,pin(collected/'frozen.json')['sha256'],manifest['limits'],'managed'))
    for index,row in enumerate(actual['records']):assert read(collected/f'process-managed-run/worker/{index:02}.json')==row
    old_manifest=read(original/'manifest.json');old_frozen=read(original/'frozen.json')
    native=read(original/'process-native-run/worker/result.json');old_managed=read(original/'process-managed-run/worker/result.json')
    assert native==read(collected/'prior/native.json') and old_managed==read(collected/'prior/managed.json')
    worker(native,old_manifest,old_frozen,original,'ort');worker(old_managed,old_manifest,old_frozen,original,'managed')
    assert manifest['cases']==old_manifest['cases'] and manifest['models']==old_manifest['models']
    comparisons=[dict(name=c['name'],**compare(a['result'],n['result'])) for c,a,n in zip(manifest['cases'],actual['records'],native['records'],strict=True)]
    prior_comparisons=[dict(name=c['name'],**compare(a['result'],n['result'])) for c,a,n in zip(manifest['cases'],actual['records'],old_managed['records'],strict=True)]
    dataset=read(collected/'inputs/dataset.json');old_scores=read(collected/'prior/scores.json');scores=[]
    keys=['reference_speaker_seconds','correct_speaker_seconds','missed_speaker_seconds','false_alarm_speaker_seconds','confused_speaker_seconds','diarization_error_rate']
    for index,case in enumerate(dataset['cases']):
        for timeline in ['intervals','exclusive_intervals']:
            hypothesis=[[s,e,int(label)] for s,e,label in actual['records'][index]['result'][timeline]]
            official=score(case['intervals'],hypothesis,600);independent=independent_score(case['intervals'],hypothesis,600)
            assert all(math.isclose(official[k],independent[k],rel_tol=1e-11,abs_tol=1e-8) for k in keys)
            prior=next(row for row in old_scores['scores'] if row['engine']=='ort' and row['name']==case['name'] and row['timeline']==timeline)
            same=all(math.isclose(official[k],prior[k],rel_tol=1e-11,abs_tol=1e-8) for k in keys)
            scores.append(dict(name=case['name'],timeline=timeline,unchanged=same,official=official,independent=independent,prior=prior))
    aggregates=[]
    for timeline in ['intervals','exclusive_intervals']:
        totals={key:math.fsum(row['official'][key] for row in scores if row['timeline']==timeline) for key in keys[:-1]}
        aggregates.append(dict(timeline=timeline,**totals,diarization_error_rate=sum(totals[k] for k in keys[2:-1])/totals['reference_speaker_seconds']))
    inputs=read(collected/'process-managed-inputs/worker/inputs.json');assert inputs['passed'] and inputs['affinity']==4
    assert inputs['cases']==[{k:c[k] for k in ['name','samples','pcm_sha256']} for c in manifest['cases']]
    evidence=dict(execution_passed=True,public_passed=all(c['passed'] for c in comparisons),accuracy_unchanged=all(s['unchanged'] for s in scores),
                  comparisons=comparisons,prior_comparisons=prior_comparisons,scores=scores,aggregates=aggregates,refusals=refusal,
                  process=dict(seconds=state['seconds'],samples=len(samples),peak_rss=state['peak_rss'],min_available=min(s['available'] for s in samples),accounting=state['accounting']),
                  product=dict(source=manifest['product_source'],core=manifest['core_sha256'],data=manifest['data_sha256'],runner=actual['runner_sha256']),
                  calls=actual['records'],setup_seconds=actual['setup_seconds'],births=receipt['births'],frozen=pin(collected/'frozen.json'))
    # A valid completed replay can close with disagreements; do not hide them behind an assertion.
    write(base/'audit.json',evidence)
    print(json.dumps(dict(execution_passed=True,public_passed=evidence['public_passed'],accuracy_unchanged=evidence['accuracy_unchanged'],comparisons=comparisons,aggregates=aggregates)))


if __name__=='__main__':main()
