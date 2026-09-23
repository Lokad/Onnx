"""Audit complete numerical census, cross-product identities, codegen and resources."""
import json
import re
from fixtures import verify_prefixes
from protocol import JOBS,LIMITS,pin,read,save,check_sample
from prepare import ROOT,BASE,previous_closed

def main():
    previous_closed();assert not (BASE/'closed.json').exists()
    prepared=read(BASE/'prepared.json')
    for name,wanted in prepared['files'].items():assert pin(ROOT/name)==wanted,name
    folder=BASE/'collected';receipt=read(folder/'collection.json');state=read(folder/'identity.json')
    assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
    assert receipt['payload']==pin(BASE/'payload.json')
    payload=read(BASE/'payload.json')
    transfer=read(BASE/'collection-transfer.json');assert transfer['passed']
    assert transfer['archive']==pin(BASE/'results.tar.gz') and transfer['receipt']==pin(folder/'collection.json')
    assert prepared['archive']==pin(BASE/'payload.tar.gz') and prepared['stage']==pin(BASE/'bundle/stage.json')
    for name,wanted in receipt['files'].items():assert pin(folder/name)==wanted,name
    assert state['complete'] and state['code']==0 and state['boot_time']==1789634288.0
    assert state['supervisor']==read(BASE/'deployment.json')
    assert state['ended']-state['started']<4*3600
    assert [r['name'] for r in state['runs']]==payload['jobs']==JOBS
    assert receipt['identities']==[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    resources=0;peak=0;identities={(v['pid'],v['birth']) for v in receipt['identities']}
    assert (state['supervisor']['pid'],state['supervisor']['birth']) in identities
    for row in state['runs']:
        assert row['complete'] and row['code']==0 and row['seconds']<LIMITS['seconds']
        limit=LIMITS['build_preflight_available' if row['name'] in JOBS[:3] else 'preflight_available']
        assert row['preflight']['available']>=limit and row['preflight']['tmpfs']>=LIMITS['preflight_tmpfs']
        samples=[json.loads(line) for line in (folder/'logs'/(row['name']+'.jsonl')).read_text().splitlines()]
        assert len(samples)==row['samples'] and samples
        for sample in samples:
            check_sample(sample)
            for member in sample['members']:
                assert (member['pid'],member['birth']) in identities
                assert row['members'][str(member['pid'])]==member['birth']
        assert max(s['rss'] for s in samples)==row['peak_rss']
        resources+=len(samples);peak=max(peak,row['peak_rss'])
    built=read(folder/'built.json');assert built['passed']
    assert (folder/'logs/sdk-version.stdout').read_text().strip().endswith('10.0.204')
    for name,wanted in built['files'].items():assert pin(folder/name)==wanted,name
    for role,files in payload['products'].items():
        for name,wanted in files.items(): assert pin(folder/'runtimes'/role/name)==wanted
    reports={}
    for run in state['runs'][3:]:
        value=read(folder/run['name']/'result.json');reports[run['name']]=value
        role=value['role'];assert role==run['name'].split('-')[0]
        assert value['runtime']=='10.0.8' and value['pid']==run['child']['pid'] and value['flags']=={}
        assert value['assembly']==built['consumer']['sha256']
        assert value['core_sha256']==payload['products'][role]['Lokad.Onnx.dll']['sha256']
    from score import score
    capture=read(BASE/'bundle/fixtures/result.json')
    assert verify_prefixes(read(BASE/'bundle/evidence/original-capture.json'),capture,BASE/'bundle/fixtures')
    verdict=score(reports,capture)
    analysis=dict(passed=True,root_product_changed=False,products=payload['products'],consumer=built['consumer'],
        resources=resources,peak_rss=peak,**verdict)
    save(BASE/'analysis.json',analysis)
    files={p.relative_to(BASE).as_posix():pin(p) for p in folder.rglob('*') if p.is_file()}
    for name in ['analysis.json','prepared.json','staged.json','payload.json','deployment.json','collection-transfer.json','results.tar.gz','payload.tar.gz']:
        files[name]=pin(BASE/name)
    save(BASE/'closed.json',dict(passed=True,admitted=verdict['admitted'],files=files,root_product_changed=False))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),resources=resources,peak_rss=peak,**verdict)))


if __name__=='__main__':main()
