"""Audit every raw observation and retained resource sample for both AMD modes."""
import json
from run import BASE,ROOT,prepared
from protocol import LIMITS,check_sample,pin,read,save


def main():
    spec=prepared();assert not (BASE/'closed.json').exists()
    payload=read(BASE/'payload/payload.json');collected=BASE/'collected';receipt=read(collected/'collection.json')
    transfer=read(BASE/'collection-transfer.json')
    assert transfer['passed'] and transfer['archive']==pin(BASE/'results.tar.gz') and transfer['receipt']==pin(collected/'collection.json')
    assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
    for name,wanted in receipt['files'].items():assert pin(collected/name)==wanted,name
    state=read(collected/'identity.json');assert state['complete'] and state['code']==0
    assert state['supervisor']==read(BASE/'deployment.json') and state['boot_time']==1789634288.0
    assert [r['name'] for r in state['runs']]==['qualify-256','qualify-512']
    assert receipt['identities']==[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    original=read(BASE/'payload/windows-256.json');resources=[];reports={}
    for row in state['runs']:
        assert row['complete'] and row['code']==0 and row['seconds']<900
        assert row['preflight']['available']>=12*1024**3 and row['preflight']['tmpfs']>=3*1024**3
        assert row['preflight']==row['preflight_observations'][-1]==read(collected/(row['name']+'-preflight.json'))[-1]
        samples=[json.loads(s) for s in (collected/'logs'/(row['name']+'.jsonl')).read_text().splitlines()]
        assert len(samples)==row['samples']>0 and max(s['rss'] for s in samples)==row['peak_rss']
        for s in samples:
            check_sample(s)
            for p in s['members']:assert row['members'][str(p['pid'])]==p['birth']
        result=read(collected/row['name']/'result.json')
        assert result['passed'] and result['geometries']==312 and result['cases']==2648 and result['layout_cases']==331
        assert result['finite_kernel_cases']==2496 and result['nonfinite_fallback_cases']==152 and result['rejected']==10 and result['owned_outputs']
        assert result['failed_cases']==result['scalar_differences']==result['production_differences']==0
        assert result['observations']==original['observations'] and result['supplemental']==original['supplemental']
        assert len(result['supplemental'])==20 and result['values']==original['values']
        assert result['runtime']=='10.0.8' and result['avx512'] and not result['flags']
        assert result['lanes']==(8 if row['name']=='qualify-256' else 16)
        assert result['core']==payload['core']['sha256'] and result['executable']==payload['consumer']['sha256'] and result['pid']==row['child']['pid']
        reports[row['name']]={k:result[k] for k in ['cases','values','finite_kernel_cases','nonfinite_fallback_cases','rejected','lanes','avx512','runtime']}
        resources.append(dict(name=row['name'],samples=len(samples),peak_rss=row['peak_rss'],seconds=row['seconds']))
    analysis=dict(passed=True,reports=reports,resources=resources,samples=sum(r['samples'] for r in resources),
        peak_rss=max(r['peak_rss'] for r in resources),no_performance_measurement=True)
    save(BASE/'analysis.json',analysis)
    files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}
    save(BASE/'closed.json',dict(passed=True,files=files,local_inputs=spec['files'],remote_terminal=receipt['identities'],analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),reports=reports,samples=analysis['samples'],peak_rss=analysis['peak_rss'])))


if __name__=='__main__':main()
