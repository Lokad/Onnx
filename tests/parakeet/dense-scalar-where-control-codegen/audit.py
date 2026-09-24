"""Audit exact diagnostic workload, outputs, clocks, identities and resources without scoring."""
import json
import re
from fixtures import screen_cases
from protocol import ORDER,JOBS,LIMITS,DIAGNOSTIC_FLAGS,pin,read,save,check_sample
from prepare import ROOT,BASE,CONTROL,CONSUMER,previous_closed

def validate_result(value, cases):
    assert value['completed'] and value['protocol']=='parakeet-dense-where-whole-census-600-180-v1'
    assert type(value['frequency']) is int and value['frequency']>0
    assert len(value['rows'])==len(cases)==220
    calls=0
    for i,(row,case) in enumerate(zip(value['rows'],cases,strict=True)):
        assert row['index']==i and row['name']==case['name'] and row['dtype']==case['dtype']
        assert all(row[k] for k in ['exact','inputs','owned','held'])
        assert row['output']==case['expected_output'] and row['shape']==case['output_shape']
        elements=__import__('math').prod(row['shape'])
        batch=max(1,min(1024,65536//max(1,elements)))
        assert row['values']==elements and row['batch']==case['batch']==batch
        assert len(row['clocks'])==780
        for j,clock in enumerate(row['clocks']):
            assert clock['index']==i and clock['name']==case['name'] and clock['batch']==batch
            assert clock['iteration']==j and clock['warmup']==(j<600)
            assert type(clock['ticks']) is int and clock['ticks']>0
        calls+=780*batch
    return calls


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
        limit=LIMITS['build_preflight_available' if row['name']=='sdk-version' else 'preflight_available']
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
    assert built['consumer']==CONSUMER and pin(folder/'built.json')==pin(CONTROL/'collected/built.json')
    assert (folder/'logs/sdk-version.stdout').read_text().strip().endswith('10.0.204')
    for name,wanted in built['files'].items():assert pin(folder/name)==wanted,name
    for role,files in payload['products'].items():
        for name,wanted in files.items(): assert pin(folder/'runtimes'/role/name)==wanted
    census=screen_cases(read(BASE/'bundle/evidence/capture-result.json'),read(BASE/'bundle/evidence/qualified-reference.json'))
    assert census==read(BASE/'bundle/cases.json') and len(census)==220
    for name,wanted in read(BASE/'bundle/stage.json')['files'].items():
        assert payload['files'][name]==wanted and pin(BASE/'bundle'/name)==wanted,name
    setup_rows=[]; public_calls=0; sample_clocks=0
    for sequence,run in enumerate(state['runs'][1:]):
        name=run['name'];role,mode,width=name.split('-');value=read(folder/name/'result.json')
        assert name==ORDER[sequence] and value['runtime']=='10.0.8'
        assert value['role']==role and value['mode']==mode and value['width']==512 and value['sequence']==sequence
        assert value['assembly']==built['consumer']['sha256'] and value['pid']==run['child']['pid']
        assert value['core_sha256']==payload['products'][role]['Lokad.Onnx.dll']['sha256']
        assert value['flags']==DIAGNOSTIC_FLAGS and value['avx512'] and value['avx2'] and value['fma']
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
        public_calls += validate_result(value,census)
        sample_clocks += len(clocks)
    for i in range(220):
        assert len({tuple(setup_rows[j*220+i]['inputs']) for j in range(4)})==1
    old_setups=read(CONTROL/'setups.json'); assert len(old_setups)==len(setup_rows)==880
    for old,new in zip(old_setups,setup_rows,strict=True):
        assert {k:v for k,v in old.items() if k!='preparation_ticks'}=={k:v for k,v in new.items() if k!='preparation_ticks'}
    assert public_calls==232929840 and sample_clocks==686400
    save(BASE/'setups.json',setup_rows)
    analysis=dict(passed=True,performance_admitted=False,products=payload['products'],consumer=built['consumer'],
        resources=resources,peak_rss=peak,setup_count=len(setup_rows), sample_clocks=sample_clocks,
        public_calls=public_calls,measured_public_calls=public_calls*180//780,
        warmup_clocks=528000,measured_clocks=158400,diagnostic_flags=DIAGNOSTIC_FLAGS,
        diagnostic_only=True,candidate_measured=False,source_root_changed=False,complete_application_result=False)
    save(BASE/'analysis.json',analysis)
    save(BASE/'closed.json',dict(passed=True,performance_admitted=False,
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()},analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),**analysis)))


if __name__=='__main__':main()
