"""Preserve the complete AVX2 control and its one-ulp diagnostic metric mismatch."""
import importlib.util
import json
import sys
from pathlib import Path
from run import BASE, ROOT, prepared
from protocol import pin, read, save, check_sample


def main():
    prepared();assert not (BASE/'failure-closed.json').exists()
    c=BASE/'collected';receipt=read(c/'collection.json');state=read(c/'identity.json')
    assert receipt['terminal'] and receipt['code']==state['code']==1 and receipt['input_error'] is None and state['complete']
    for name,wanted in receipt['files'].items():assert pin(c/name)==wanted,name
    assert state['supervisor']==read(BASE/'deployment.json')
    assert [r['name'] for r in state['runs']]==['consumer-restore','consumer-build','selected-256']
    assert receipt['identities']==[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    resources=[]
    for r in state['runs']:
        assert r['complete'] and r['code']==0 and r['seconds']<900
        assert r['preflight']['available']>=12*1024**3 and r['preflight']['tmpfs']>=3*1024**3
        samples=[json.loads(s) for s in (c/'logs'/(r['name']+'.jsonl')).read_text().splitlines()]
        assert len(samples)==r['samples'] and max(s['rss'] for s in samples)==r['peak_rss']
        for s in samples:
            check_sample(s)
            assert all(r['members'][str(m['pid'])]==m['birth'] for m in s['members'])
        resources.append(dict(name=r['name'],samples=len(samples),peak_rss=r['peak_rss'],seconds=r['seconds']))
    payload=read(BASE/'payload/payload.json');built=read(c/'built.json');payload['consumer']=built['consumer']
    for name,wanted in built['files'].items():
        if name.startswith('built/'):assert pin(c/name)==wanted,name
    spec=importlib.util.spec_from_file_location('platform_metric_successor',ROOT/'tests/pyannote/lstm-platform-reference-amd-v3/checks.py')
    checks=importlib.util.module_from_spec(spec);spec.loader.exec_module(checks)
    sys.path.insert(0,str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
    report=checks.check_result(read(c/'selected-256/result.json'),'selected','256',payload,BASE/'payload',c/'selected-256')
    assert report['windows_changed']==2760172 and report['windows_maximum']==1.3287595951668142e-6 and report['maximum']==1.591294694046004e-5
    files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}
    save(BASE/'failure-closed.json',dict(passed=False,closed_failure=True,reason='All selected AVX2 calls completed. Two repeat observations report Windows diagnostic maximum one double ULP below independently recalculated value; exact metadata check stopped remaining jobs. No output tolerance failure.',files=files,remote_terminal=receipt['identities'],resources=resources,retained_control=report,retained_consumer=built['consumer'],no_performance_measurement=True))
    print(json.dumps(dict(closed=pin(BASE/'failure-closed.json'),resources=resources,report=report)))


if __name__=='__main__':main()
