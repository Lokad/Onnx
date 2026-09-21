"""Audit complete meeting requests against the retained native public results."""
import json
from pathlib import Path
import sys
from prepare import ROOT, BASE, TOOLS, pin, save
from run import absent
sys.path.insert(0,str(ROOT/'tests/pyannote/natural-meetings'))
from audit import compare, inspect_public


def main():
    target=BASE/'closed.json';assert not target.exists()
    prepared=json.loads((BASE/'prepared.json').read_text());controls=json.loads((BASE/'run-controls.json').read_text())
    assert controls['prepared']==pin(BASE/'prepared.json') and controls['runner']==pin(TOOLS/'run.py')
    for name,wanted in prepared['files'].items():assert pin(ROOT/name)==wanted,name
    state=json.loads((BASE/'processes.json').read_text());manifest=json.loads((BASE/'manifest.json').read_text());limits=manifest['limits']
    assert prepared['manifest']==pin(BASE/'manifest.json') and controls['limits']==limits
    assert state['complete'] and state['code']==0 and absent(state['supervisor'])
    assert [r['mode'] for r in state['runs']]==['inputs','run']
    identities=[state['supervisor']];samples=0
    for run in state['runs']:
        assert run['complete'] and run['code']==0 and absent(run['worker']);identities.append(run['worker'])
        folder=BASE/'process'/run['mode']
        assert not (folder/'stderr.txt').read_text().strip()
        resource=[json.loads(s) for s in (folder/'samples.jsonl').read_text().splitlines()]
        assert len(resource)==run['samples']>0 and max(s['rss'] for s in resource)==run['peak_rss']
        assert resource[-1]['seconds']<=run['seconds']<limits['seconds'];samples+=len(resource)
        for s in resource:
            assert s['rss']<limits['rss'] and s['available']>=limits['available'] and s['disk']>=limits['disk']
            assert s['affinity']==[2] and s['pid']==run['worker']['pid'] and s['birth']==run['worker']['birth']
        gaps=[resource[0]['seconds']]+[b['seconds']-a['seconds'] for a,b in zip(resource,resource[1:])]+[run['seconds']-resource[-1]['seconds']]
        assert all(0<=v<10 for v in gaps)
        preflight=[json.loads(s) for s in (folder/'preflight.jsonl').read_text().splitlines()]
        assert preflight[-1]==run['preflight'] and preflight[-1]['available']>=limits['preflight']
        assert all(s['disk']>=limits['disk'] and s['seconds']<limits['preflight_wait_seconds'] for s in preflight)
    inputs=json.loads((BASE/'output-inputs/inputs.json').read_text())
    assert inputs['passed'] and inputs['affinity']==4
    assert inputs['cases']==[{k:c[k] for k in ['name','samples','pcm_sha256']} for c in manifest['cases']]
    result=json.loads((BASE/'output-run/result.json').read_text())
    assert result['schema']==1 and result['engine']=='managed' and result['held_outputs_unchanged']
    assert result['runtime']=='.NET 10.0.12' and result['affinity']==4 and not result['flags']
    assert result['manifest_sha256']==pin(BASE/'manifest.json')['sha256']
    for key,file in [('core_sha256','Lokad.Onnx.dll'),('data_sha256','Lokad.Onnx.Data.dll'),('runner_sha256','NaturalMeetings.dll')]:
        assert result[key]==pin(BASE/'bin'/file)['sha256']
    native=json.loads((BASE/'prior/native.json').read_text());prior_manifest=json.loads((BASE/'prior/manifest.json').read_text())
    assert manifest['cases']==prior_manifest['cases'] and manifest['models']==prior_manifest['models']
    assert [r['name'] for r in result['records']]==[c['name'] for c in manifest['cases']]
    comparisons=[];exact_timelines=True
    for i,(row,wanted,case) in enumerate(zip(result['records'],native['records'],manifest['cases'],strict=True)):
        assert row==json.loads((BASE/'output-run'/f'{i:02}.json').read_text())
        assert row['name']==wanted['name']==case['name'] and row['input_sha256']==case['pcm_sha256'] and row['ownership']
        assert row['seconds']==(row['end_ticks']-row['start_ticks'])/row['frequency']>0
        inspect_public(row['result'],case['samples']);inspect_public(wanted['result'],case['samples'])
        same=all(row['result'][key]==wanted['result'][key] for key in ['intervals','exclusive_intervals'])
        exact_timelines=exact_timelines and same
        comparisons.append(dict(name=case['name'],seconds=row['seconds'],exact_native_timelines=same,**compare(row['result'],wanted['result'])))
    old_scores=json.loads((BASE/'prior/scores.json').read_text())
    scores=[r for r in old_scores['scores'] if r['engine']=='ort'] if exact_timelines else []
    aggregates=[r for r in old_scores['aggregates'] if r['engine']=='ort'] if exact_timelines else []
    assert not exact_timelines or len(scores)==4 and len(aggregates)==2
    passed=all(c['passed'] for c in comparisons)
    analysis=dict(passed=passed,comparisons=comparisons,exact_native_timelines=exact_timelines,
        scores=scores,aggregates=aggregates,score_scope='Retained native scores apply only when both complete timelines are exactly identical',
        samples=samples,peak_rss=max(r['peak_rss'] for r in state['runs']),identities=identities,
        limitation='Correctness replay, no new native inference or matched performance claim')
    save(BASE/'analysis.json',analysis)
    files=dict(prepared['files'])
    for path in BASE.rglob('*'):
        if path.is_file():files[str(path.relative_to(ROOT))]=pin(path)
    for path in TOOLS.iterdir():
        if path.is_file():files[str(path.relative_to(ROOT))]=pin(path)
    save(target,dict(passed=passed,files=files,identities=identities,analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(passed=passed,comparisons=comparisons,exact_native_timelines=exact_timelines,samples=samples,closure=pin(target))))
    assert passed,'Meeting incompatibilities retained'


if __name__=='__main__':main()
