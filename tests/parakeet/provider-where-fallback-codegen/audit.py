"""Audit untimed case execution, code-generation presence and resources."""
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
        gaps=[samples[0]['seconds']]+[b['seconds']-a['seconds'] for a,b in zip(samples,samples[1:])]+[row['seconds']-samples[-1]['seconds']]
        assert all(0<=gap<10 for gap in gaps)
        resources+=len(samples);peak=max(peak,row['peak_rss'])
    built=read(folder/'built.json');assert built['passed']
    assert (folder/'logs/sdk-version.stdout').read_text().strip().endswith('10.0.204')
    for name,wanted in built['files'].items():assert pin(folder/name)==wanted,name
    for role,files in payload['products'].items():
        for name,wanted in files.items(): assert pin(folder/'runtimes'/role/name)==wanted
    census=screen_cases(read(BASE/'bundle/evidence/capture-result.json'),read(BASE/'bundle/evidence/qualified-reference.json'))
    assert census==read(BASE/'bundle/cases.json') and len(census)==122
    for name,wanted in read(BASE/'bundle/stage.json')['files'].items():
        assert payload['files'][name]==wanted and pin(BASE/'bundle'/name)==wanted,name
    results={};calls=0;bodies={}
    for run in state['runs'][3:]:
        name=run['name'];role,mode,width=name.split('-');value=read(folder/name/'result.json')
        assert value['completed'] and value['no_performance_measurement'] and value['protocol']=='parakeet-provider-where-fallback-codegen-v1'
        assert value['runtime']=='10.0.8' and value['role']==role and value['mode']=='diagnostic' and value['width']==512
        assert value['assembly']==built['consumer']['sha256'] and value['pid']==run['child']['pid']
        assert value['core_sha256']==payload['products'][role]['Lokad.Onnx.dll']['sha256']
        assert value['flags']=={'DOTNET_JitDisasm':'Lokad.Onnx.CPUExecutionProvider:Where Lokad.Onnx.Tensor`1:Where Lokad.Onnx.UniformScalarWhere:Try Program:Exercise'}
        assert value['avx512'] and value['avx2'] and value['fma']
        journal=[json.loads(line) for line in (folder/name/'cases.jsonl').read_text().splitlines()]
        assert len(value['rows'])==122 and len(journal)==244
        for i,(row,case) in enumerate(zip(value['rows'],census,strict=True)):
            assert row['index']==i and row['name']==case['name'] and row['dtype']==case['dtype']
            assert row['shape']==case['output_shape'] and row['output']==case['expected_output']
            assert row['values']==int(np.prod(case['output_shape'])) and row['batch']==case['batch']
            assert row['calls']==120*case['batch'] and all(row[k] for k in ['exact','inputs','owned','held','provider_metadata'])
            assert journal[2*i]['phase']=='prepared';setup=journal[2*i]['setup']
            for key in ['index','name','dtype','shape','values','batch','output']:assert setup[key]==row[key]
            assert len(setup['inputs'])==3 and all(re.fullmatch('[0-9a-f]{64}',h) for h in setup['inputs'])
            assert journal[2*i+1]==dict(phase='complete',index=i,name=case['name'],calls=row['calls'])
            calls+=row['calls']
        disasm=(folder/'logs'/(name+'.stdout')).read_text()
        assert 'Assembly listing for method Lokad.Onnx.Tensor`1[float]:Where' in disasm
        assert 'Assembly listing for method Program:Exercise[float]' in disasm
        assert 'Assembly listing for method Lokad.Onnx.CPUExecutionProvider:Where' in disasm
        if role=='candidate':assert 'Assembly listing for method Lokad.Onnx.UniformScalarWhere:Try[float]' in disasm
        bodies[role]=dict(stdout=pin(folder/'logs'/(name+'.stdout')),headers=disasm.count('; Assembly listing for method'))
        results[role]=value['rows']
    assert results['current']==results['candidate'] and calls==17640480
    analysis=dict(passed=True,no_performance_measurement=True,products=payload['products'],consumer=built['consumer'],
        workers=2,cases_per_worker=122,calls=calls,all_selected_bits_exact=True,input_ownership_and_held_outputs=True,
        codegen=bodies,resources=resources,peak_rss=peak,root_product_changed=False,
        prior_screen_remains_rejected=True,generated_code_review_pending=True)
    save(BASE/'analysis.json',analysis)
    save(BASE/'closed.json',dict(passed=True,files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()},analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),**analysis)))


if __name__=='__main__':main()
