"""Audit the fixed performance census, immutable identities, journals and resources."""
import json
import re
import numpy as np
from fixtures import screen_cases
from protocol import ORDER,JOBS,LIMITS,pin,read,save,check_sample
from prepare import ROOT,BASE,previous_closed

def main():
    previous_closed();assert not (BASE/'closed.json').exists()
    prepared=read(BASE/'prepared.json')
    for name,wanted in prepared['files'].items():assert pin(ROOT/name)==wanted,name
    folder=BASE/'collected';receipt=read(folder/'collection.json');state=read(folder/'identity.json')
    assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
    assert receipt['payload']==pin(BASE/'payload.json')
    payload=read(BASE/'payload.json'); assert payload['products']['current']==payload['products']['candidate']
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
        gaps=[samples[0]['seconds']]+[b['seconds']-a['seconds'] for a,b in zip(samples,samples[1:])]+[row['seconds']-samples[-1]['seconds']]
        assert all(0<=gap<10 for gap in gaps)
        resources+=len(samples);peak=max(peak,row['peak_rss'])
    built=read(folder/'built.json');assert built['passed']
    assert (folder/'logs/sdk-version.stdout').read_text().strip().endswith('10.0.204')
    for name,wanted in built['files'].items():assert pin(folder/name)==wanted,name
    for role,files in payload['products'].items():
        for name,wanted in files.items(): assert pin(folder/'runtimes'/role/name)==wanted
    census=screen_cases(read(BASE/'bundle/evidence/capture-result.json'),read(BASE/'bundle/evidence/qualified-reference.json'))
    assert census==read(BASE/'bundle/cases.json') and len(census)==220
    for name,wanted in read(BASE/'bundle/stage.json')['files'].items():
        assert payload['files'][name]==wanted and pin(BASE/'bundle'/name)==wanted,name
    from score import score
    reports={};setup_rows=[];clock_rows=[]
    for sequence,run in enumerate(state['runs'][3:]):
        name=run['name'];role,mode,width=name.split('-');value=read(folder/name/'result.json')
        assert name==ORDER[sequence] and value['runtime']=='10.0.8'
        assert value['role']==role and value['mode']==mode and value['width']==512 and value['sequence']==sequence
        assert value['assembly']==built['consumer']['sha256'] and value['pid']==run['child']['pid']
        assert value['core_sha256']==payload['products'][role]['Lokad.Onnx.dll']['sha256']
        assert value['flags']=={} and value['avx512'] and value['avx2'] and value['fma']
        clocks=[json.loads(line) for line in (folder/name/'clocks.jsonl').read_text().splitlines()]
        setups=[json.loads(line) for line in (folder/name/'setups.jsonl').read_text().splitlines()]
        assert clocks==[clock for first,end in [(0,600),(600,780)] for row in value['rows'] for clock in row['clocks'][first:end]]
        assert len(setups)==220 and len(clocks)==171600
        for i,(setup,row) in enumerate(zip(setups,value['rows'],strict=True)):
            assert setup['index']==i
            for key in ['name','dtype','batch','shape','values','output']:assert setup[key]==row[key]
            assert type(setup['preparation_ticks']) is int and setup['preparation_ticks']>0
            assert len(setup['inputs'])==3 and all(re.fullmatch('[0-9a-f]{64}',v) for v in setup['inputs'])
            setup_rows.append(dict(process=name,**setup))
        clock_rows.extend(dict(process=name,frequency=value['frequency'],**clock) for clock in clocks)
        reports[name]=value
    for i in range(220):
        assert len({tuple(setup_rows[j*220+i]['inputs']) for j in range(4)})==1
    result=score(reports,census)
    import csv
    with (BASE/'clocks.csv').open('x',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(clock_rows[0]));writer.writeheader();writer.writerows(clock_rows)
    with (BASE/'setups.json').open('x',encoding='utf8') as stream:json.dump(setup_rows,stream,indent=2)
    analysis=dict(passed=True,stability_admitted=result['admitted'],products=payload['products'],consumer=built['consumer'],
        comparison=result,resources=resources,peak_rss=peak,setup_count=len(setup_rows),
        identical_binary_control=True,candidate_measured=False,source_root_changed=False,complete_application_result=False)
    save(BASE/'analysis.json',analysis)
    save(BASE/'closed.json',dict(passed=True,stability_admitted=result['admitted'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()},analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),stability_admitted=result['admitted'],
        scopes=result['scopes'],controls_failed=sum(not c['passed'] for c in result['controls']),gates=result['gates'],
        resources=resources,peak_rss=peak,public_calls=result['public_calls'])))


if __name__=='__main__':main()
