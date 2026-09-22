"""Independently reconcile every numerical result, resource sample and terminal owner."""
import json
import importlib.util
from run import BASE, ROOT, prepared
from prepare import monitor
from protocol import JOBS, LIMITS, check_sample, pin, read, save
from checks import check_result, check_suite, check_native


def main():
    spec = prepared(); assert not (BASE/'closed.json').exists()
    payload = read(BASE/'payload/payload.json'); collected = BASE/'collected'; receipt = read(collected/'collection.json')
    transfer = read(BASE/'collection-transfer.json')
    assert transfer['passed'] and transfer['archive'] == pin(BASE/'results.tar.gz') and transfer['receipt'] == pin(collected/'collection.json')
    assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
    for name, wanted in receipt['files'].items(): assert pin(collected/name) == wanted, name
    for name, wanted in payload['files'].items(): assert pin(BASE/'payload'/name) == wanted, name
    state = read(collected/'identity.json'); assert state['complete'] and state['code'] == 0
    assert state['supervisor'] == read(BASE/'deployment.json') and state['boot_time'] == 1789634288.0
    assert [r['name'] for r in state['runs']] == payload['jobs'] == JOBS
    assert receipt['identities'] == [state['supervisor']]+[dict(pid=int(p), birth=b) for r in state['runs'] for p,b in r['members'].items()]
    built=read(collected/'built.json');assert built['passed'];payload['consumer']=built['consumer']
    for file,wanted in built['files'].items():
        if file.startswith('built/'):assert pin(collected/file)==wanted,file
    assert built['consumer']==pin(collected/'built/LstmModelReplay.dll')
    resources = []; reports = {}
    for row in state['runs']:
        mode, width = row['name'].split('-')
        assert row['complete'] and row['code'] == 0 and row['seconds'] < LIMITS['seconds']
        assert row['preflight']['available'] >= LIMITS['preflight_available'] and row['preflight']['tmpfs'] >= LIMITS['preflight_tmpfs']
        assert row['preflight'] == row['preflight_observations'][-1] == read(collected/(row['name']+'-preflight.json'))[-1]
        samples = [json.loads(s) for s in (collected/'logs'/(row['name']+'.jsonl')).read_text().splitlines()]
        assert len(samples) == row['samples'] > 0 and max(s['rss'] for s in samples) == row['peak_rss']
        for sample in samples:
            check_sample(sample)
            for member in sample['members']: assert row['members'][str(member['pid'])] == member['birth']
        if mode=='consumer':pass
        elif mode=='suite':reports[row['name']]=check_suite(collected/row['name']/'suite.trx',BASE/'payload/ordinary.trx')
        else:
            result=read(collected/row['name']/'result.json');assert result['pid']==row['child']['pid']
            if mode=='native':reports[row['name']]=check_native(result,collected/row['name'],BASE/'payload')
            else:
                assert result['runtime']=='10.0.8'
                reports[row['name']]=check_result(result,mode,width,payload,BASE/'payload',collected/row['name'])
        resources.append(dict(name=row['name'], samples=len(samples), peak_rss=row['peak_rss'], seconds=row['seconds']))
    assert state['ended']-state['started'] < 4*3600
    import numpy as np
    native=read(collected/'native-ort/result.json');capture=read(BASE/'payload/fixtures/output/result.json')
    native_comparisons={};width_agreement={}
    for width in ['256','512','scalar']:
        maximum=0.;count=0
        for i,row in enumerate(native['reports']):
            a=np.fromfile(collected/('selected-'+width)/row['file'],dtype='<f4').astype('float64')
            b=np.fromfile(collected/'native-ort'/row['file'],dtype='<f4').astype('float64')
            assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            error=float((np.abs(a-b)/np.maximum(1.,np.abs(b))).max(initial=0));assert error<=1e-4
            maximum=max(maximum,error);count+=int(a.size)
        assert count==1815552
        native_comparisons[width]=dict(values=count,maximum=maximum)
        width_agreement[width]=all(pin(collected/('selected-'+width)/row['file'])==pin(collected/'selected-256'/row['file']) for row in native['reports'])

    analysis = dict(passed=True, reports=reports, resources=resources,
        samples=sum(r['samples'] for r in resources), peak_rss=max(r['peak_rss'] for r in resources),
        cores=payload['cores'], consumer=payload['consumer'], no_performance_measurement=True, cross_platform_diagnostic=True, no_local_worker_due_disk_preflight=True, native_comparisons=native_comparisons, width_agreement=width_agreement, candidate_admission_pending=True)
    save(BASE/'analysis.json', analysis)
    files = {p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file() and not {'obj','packages'}.intersection(p.relative_to(BASE).parts)}
    save(BASE/'closed.json', dict(passed=True, files=files, local_inputs=spec['files'], remote_terminal=receipt['identities'], analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'), **analysis)))


if __name__ == '__main__': main()
