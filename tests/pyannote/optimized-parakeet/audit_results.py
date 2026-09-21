"""Separate preserved native failures from regression of complete Parakeet trajectories."""
import json
from pathlib import Path
import sys
import numpy as np
from prepare import ROOT, BASE, REFERENCE, TOOLS, pin, save
from run import absent
sys.path.insert(0,str(ROOT/'tests/parakeet/transcribe'))
from audit import audit as native_audit


def main():
    target=BASE/'closed.json';assert not target.exists()
    spec=json.loads((BASE/'manifest.json').read_text());controls=json.loads((BASE/'run-controls.json').read_text());limits=spec['limits']
    assert controls['manifest']==pin(BASE/'manifest.json') and controls['runner']==pin(TOOLS/'run.py') and controls['limits']==limits
    for name,wanted in spec['files'].items():assert pin(ROOT/name)==wanted,name
    state=json.loads((BASE/'processes.json').read_text())
    assert state['complete'] and state['code']==0 and absent(state['supervisor'])
    assert [r['role'] for r in state['runs']]==spec['jobs']
    identities=[state['supervisor']];samples=0;reports={};results={}
    for run in state['runs']:
        role=run['role'];assert run['complete'] and run['code'] in [0,1] and absent(run['worker']);identities.append(run['worker'])
        result=json.loads((BASE/(role+'.json')).read_text());results[role]=result
        assert result['core_sha256']==spec['cores'][role]['sha256']
        assert result['data_sha256']=='e7fe1668e3aa08fb07b1e5a687ef2b1e4af54567f6a458db09d411eb69f99aeb'
        assert result['runner_sha256']=='335ca09d0e45e344068c484c92af9d0db43a6ae0accd1895d7ae7bb88b0afcf9'
        assert result['runtime']=='.NET 10.0.12' and not result['settings']
        report=native_audit(REFERENCE,BASE/(role+'.json'));reports[role]=report
        assert report['audit_consistent'] and report['application_passed'] and run['code']==(0 if report['numeric_gate_passed'] else 1)
        assert report['numeric_gate_passed']==run['native_numeric_passed']
        save(BASE/(role+'-native-audit.json'),report)
        folder=BASE/'process'/role;assert not (folder/'stderr.txt').read_text().strip()
        resource=[json.loads(s) for s in (folder/'samples.jsonl').read_text().splitlines()]
        assert len(resource)==run['samples']>0 and max(s['rss'] for s in resource)==run['peak_rss'];samples+=len(resource)
        assert resource[-1]['seconds']<=run['seconds']<limits['seconds']
        for s in resource:
            assert s['rss']<limits['rss'] and s['available']>=limits['available'] and s['disk']>=limits['disk']
            assert s['affinity']==[2] and s['pid']==run['worker']['pid'] and s['birth']==run['worker']['birth']
        gaps=[resource[0]['seconds']]+[b['seconds']-a['seconds'] for a,b in zip(resource,resource[1:])]+[run['seconds']-resource[-1]['seconds']]
        assert all(0<=v<10 for v in gaps)
        preflight=[json.loads(s) for s in (folder/'preflight.jsonl').read_text().splitlines()]
        assert preflight[-1]==run['preflight'] and preflight[-1]['available']>=limits['preflight']
        assert all(s['seconds']<limits['preflight_wait_seconds'] and s['disk']>=limits['disk'] for s in preflight)
    comparisons=[];values=0
    for a,b in zip(results['candidate']['rows'],results['baseline']['rows'],strict=True):
        assert a['name']==b['name'] and a['actual']==b['actual']
        for x,y in zip(a['comparisons'],b['comparisons'],strict=True):
            assert all(x[k]==y[k] for k in ['label','output','shape','dtype','file'])
            dtype={'Float':'<f4','Int32':'<i4','Int64':'<i8'}[x['dtype']]
            left=np.fromfile(BASE/'candidate.json.tensors'/x['file'],dtype=dtype)
            right=np.fromfile(BASE/'baseline.json.tensors'/y['file'],dtype=dtype)
            assert left.shape==right.shape and np.isfinite(left).all() and np.isfinite(right).all();values+=left.size
            if x['dtype']=='Float':
                delta=np.abs(left.astype(np.float64)-right.astype(np.float64))/np.maximum(1,np.abs(right.astype(np.float64)))
                error=float(delta.max(initial=0));failed=int(np.count_nonzero(delta>1e-4))
            else:error=0. if np.array_equal(left,right) else 1.;failed=int(np.count_nonzero(left!=right))
            comparisons.append(dict(case=a['name'],label=x['label'],output=x['output'],values=left.size,
                bit_identical=left.tobytes()==right.tobytes(),maximum_scaled_error=error,failed_values=failed))
    assert len(comparisons)==784 and values==3090494
    def failures(role):return {'/'.join(r[k] for k in ['case','label','output']):r for r in reports[role]['failures']}
    before=failures('baseline');after=failures('candidate')
    baseline_reproduced=set(before)==set(spec['known_native_failures'])
    no_new_failures=set(after)<=set(before)
    no_worse_failures=no_new_failures and all(r['maximum']<=before[key]['maximum'] for key,r in after.items())
    regression_passed=baseline_reproduced and no_new_failures and no_worse_failures and all(c['failed_values']==0 for c in comparisons)
    analysis=dict(regression_passed=regression_passed,baseline_failures_reproduced=baseline_reproduced,no_new_native_failures=no_new_failures,
        no_worse_existing_failures=no_worse_failures,native=reports,comparisons=comparisons,arrays=784,values=values,
        bit_identical_arrays=sum(c['bit_identical'] for c in comparisons),maximum_baseline_relative_error=max(c['maximum_scaled_error'] for c in comparisons),
        samples=samples,peak_rss=max(r['peak_rss'] for r in state['runs']),identities=identities,
        limitation='Affected-model regression; preserve full native numerical verdict separately; no matched performance claim')
    save(BASE/'analysis.json',analysis)
    files=dict(spec['files'])
    for path in BASE.rglob('*'):
        if path.is_file():files[str(path.relative_to(ROOT))]=pin(path)
    for path in TOOLS.iterdir():
        if path.is_file():files[str(path.relative_to(ROOT))]=pin(path)
    save(target,dict(regression_passed=regression_passed,files=files,identities=identities,analysis=pin(BASE/'analysis.json'),
        candidate_native_numeric_passed=reports['candidate']['numeric_gate_passed']))
    print(json.dumps({k:v for k,v in analysis.items() if k not in ['comparisons','native']}))
    print(json.dumps(dict(native_failures={k:v['failures'] for k,v in reports.items()},closure=pin(target),files=len(files))))
    assert regression_passed,'Complete regression diagnostics retained'


if __name__=='__main__':main()
