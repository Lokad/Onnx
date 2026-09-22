"""Retain the two passing AMD suites and failed unchanged-product reference check."""
import json
from run import BASE, prepared
from protocol import LIMITS, check_sample, pin, read, save
from checks import check_suite


def main():
    spec=prepared();assert not (BASE/'failure-closed.json').exists()
    collected=BASE/'collected';receipt=read(collected/'collection.json');state=read(collected/'identity.json')
    assert receipt['terminal'] and receipt['code']==state['code']==1 and receipt['input_error'] is None and state['complete']
    assert receipt['payload']==pin(BASE/'payload/payload.json') and state['supervisor']==read(BASE/'deployment.json')
    for name,wanted in receipt['files'].items():assert pin(collected/name)==wanted,name
    assert [r['name'] for r in state['runs']]==['suite-256','suite-512','selected-256']
    assert receipt['identities']==[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    resources=[]
    for r in state['runs']:
        assert r['complete'] and r['code']==(-6 if r['name']=='selected-256' else 0)
        assert r['seconds']<900 and r['preflight']['available']>=LIMITS['preflight_available'] and r['preflight']['tmpfs']>=LIMITS['preflight_tmpfs']
        rows=[json.loads(s) for s in (collected/'logs'/(r['name']+'.jsonl')).read_text().splitlines()]
        assert len(rows)==r['samples'] and max(v['rss'] for v in rows)==r['peak_rss']
        for v in rows:
            check_sample(v)
            assert all(r['members'][str(m['pid'])]==m['birth'] for m in v['members'])
        resources.append(dict(name=r['name'],samples=len(rows),peak_rss=r['peak_rss'],seconds=r['seconds']))
    suites={width:check_suite(collected/('suite-'+width)/'suite.trx',BASE/'payload/ordinary.trx') for width in ['256','512']}
    assert 'Selected full output bits' in (collected/'logs/selected-256.stderr').read_text()
    assert not list((collected/'selected-256').glob('result.json'))
    assert "NameError: name 'JOBS' is not defined" in (BASE/'collection.stderr').read_text()
    transfer=read(BASE/'collection-transfer.json');assert transfer['passed'] and transfer['archive']==pin(BASE/'results-collected.tar.gz')
    analysis=dict(closed_failure=True,passed=False,suites=suites,resources=resources,
        reason='Unchanged selected AMD/Linux runtime fails exact Windows-reference bits before a candidate model replay. No tensor arrays were emitted. Quantify platform differences before candidate admission; native tolerance remains 1e-4.',
        no_performance_measurement=True,no_candidate_model_replay=True)
    save(BASE/'failure-analysis.json',analysis)
    files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}
    save(BASE/'failure-closed.json',dict(**analysis,files=files,remote_terminal=receipt['identities'],local_inputs=spec['files']))
    print(json.dumps(dict(closed=pin(BASE/'failure-closed.json'),**analysis)))


if __name__=='__main__':main()
