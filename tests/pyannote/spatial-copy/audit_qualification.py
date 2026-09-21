"""Verify complete shared replay arrays after the candidate's broader suites."""
import json
from pathlib import Path
import numpy as np
from prepare import ROOT, BASE, pin, save
from run import absent
from audit import scaled


def main():
    target=BASE/'qualification-audit.json';assert not target.exists()
    spec=json.loads((BASE/'manifest.json').read_text());state=json.loads((BASE/'qualification.json').read_text())
    assert state['complete'] and state['code']==0 and absent(state['supervisor']) and all(s['code']==0 for s in state['steps'])
    assert [m['mode'] for m in state['models']]==['e5','shared']
    old=ROOT/'artifacts/e5-profiler-shared-v2-20260921'
    assert pin(old/'closed.json')['sha256']=='77f2ab0b4761b09be7225b69667e071a05826fb0c053a67b1e6d2a93a99ea533'
    pins=json.loads((old/'closed.json').read_text())['files']
    rows=[];samples=0;values=0
    for worker in state['models']:
        assert worker['complete'] and worker['code']==0 and absent(worker['worker'])
        mode=worker['mode'];folder=BASE/'shared-output'/mode;result=json.loads((folder/'result.json').read_text())
        assert result['passed'] and result['mode']==mode and result['enabled'] is False
        assert result['core_sha256']==spec['cores']['candidate']['sha256']
        assert result['probe_sha256']=='a50d3e965cf480844559b1f5856e3de318b5c8a269742a9e6504afb9cee6c8c0'
        assert result['inputs_unchanged'] and result['held_outputs_unchanged']
        assert result['flags']==dict(LOKAD_ONNX_FINGERPRINT_STRINGS='0') and result['runtime']=='10.0.12'
        assert len(result['graphs'])==(80 if mode=='e5' else 11) and all(g['entries']==0 for g in result['graphs'])
        previous=old/'outputs'/(mode+'-0-baseline')/'result.json'
        key=previous.relative_to(ROOT).as_posix();assert pin(previous)==pins[key]
        wanted=json.loads(previous.read_text())
        assert len(result['rows'])==len(wanted['rows'])==(60 if mode=='e5' else 106)
        reference=ROOT/('artifacts/e5-randomized-processes-20260921/payload/inputs' if mode=='e5' else 'artifacts/shared-regression-20260918/reference')
        for a,b in zip(result['rows'],wanted['rows'],strict=True):
            assert all(a[k]==b[k] for k in ('model','scenario','step','name','shape','reference_file','reference_sha256','values'))
            path=folder/a['file'];assert pin(path)['sha256']==a['sha256']
            native=reference/a['reference_file'];assert pin(native)['sha256']==a['reference_sha256']
            actual=np.fromfile(path,dtype='<f4').reshape(a['shape'])
            expected=np.fromfile(native,dtype='<f4').reshape(a['shape']) if mode=='e5' else np.load(native,allow_pickle=False)
            check=scaled(actual,expected);assert check['failed_values']==a['failed_values']==0
            assert abs(check['max_scaled_error']-a['max_scaled_error'])<1e-15
            if mode=='e5':assert a['sha256']==b['sha256'],'e5 output changed'
            rows.append(dict(mode=mode,model=a['model'],name=a['name'],**check));values+=actual.size
        assert {p.name for p in folder.iterdir()}=={'result.json'}|{r['file'] for r in result['rows']}
        resource=[json.loads(line) for line in (BASE/'shared-process'/mode/'samples.jsonl').read_text().splitlines()]
        assert resource;samples+=len(resource)
        for s in resource:
            assert s['seconds']<900 and s['rss']<8*1024**3 and s['available']>=1024**3 and s['disk']>=20*1024**3 and s['affinity']==[2]
        preflight=[json.loads(line) for line in (BASE/'shared-process'/mode/'preflight.jsonl').read_text().splitlines()]
        assert preflight[-1]['available']>=10*1024**3 and all(s['seconds']<3600 and s['disk']>=20*1024**3 for s in preflight)
    assert len(rows)==166 and values==5000814
    report=dict(passed=True,arrays=len(rows),values=values,rows=rows,samples=samples,identities=[state['supervisor']]+[m['worker'] for m in state['models']],
        maximum_native_error=max(r['max_scaled_error'] for r in rows),qualification=pin(BASE/'qualification.json'))
    save(target,report)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}))


if __name__=='__main__':main()
