"""Audit complete numerical census, cross-product identities, codegen and resources."""
import json
import re
import numpy as np
from fixtures import cases
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
        gaps=[samples[0]['seconds']]+[b['seconds']-a['seconds'] for a,b in zip(samples,samples[1:])]+[row['seconds']-samples[-1]['seconds']]
        assert all(0<=gap<10 for gap in gaps)
        resources+=len(samples);peak=max(peak,row['peak_rss'])
    built=read(folder/'built.json');assert built['passed']
    assert (folder/'logs/sdk-version.stdout').read_text().strip().endswith('10.0.204')
    for name,wanted in built['files'].items():assert pin(folder/name)==wanted,name
    for role,files in payload['products'].items():
        for name,wanted in files.items(): assert pin(folder/'runtimes'/role/name)==wanted
    census=cases(read(BASE/'bundle/evidence/capture-result.json'))
    assert census==read(BASE/'bundle/cases.json') and len(census)==203
    for name,wanted in read(BASE/'bundle/stage.json')['files'].items():
        assert payload['files'][name]==wanted and pin(BASE/'bundle'/name)==wanted,name
    import hashlib
    results={};contracts={};values=0;admitted=0;stages={}
    bits=np.array([0,0x80000000,0x7f800000,0xff800000,0x7fc12345,0xffc54321,0x7f812345,1,0x807fffff,0x3f800000,0xc61c4000],dtype='<u4')
    prior=read(BASE/'bundle/evidence/numerical-reference.json')
    prior_rows={r['name']:r for r in prior['rows']}
    for run in state['runs'][3:]:
        name=run['name'];role,mode,width=name.split('-');value=read(folder/name/'result.json')
        assert value['completed'] and value['no_performance_measurement'] and value['runtime']=='10.0.8'
        assert value['role']==role and value['mode']==mode and value['width']==int(width)
        assert value['assembly']==built['consumer']['sha256'] and value['pid']==run['child']['pid']
        assert value['core_sha256']==payload['products'][role]['Lokad.Onnx.dll']['sha256']
        flags={} if width=='512' else {'DOTNET_EnableAVX512':'0'}
        if mode=='codegen':flags['DOTNET_JitDisasm']='Lokad.Onnx.CPUExecutionProvider:Where Lokad.Onnx.Tensor`1:Where Lokad.Onnx.UniformScalarWhere:Try Program:Invoke'
        assert value['flags']==flags and value['avx512']==(width=='512')
        expected=census if mode=='numerics' else [c for c in census if not c.get('error')]
        assert len(expected)==(203 if mode=='numerics' else 188) and set(value['rows'])=={'tensor','provider'}
        clean_boundaries={}
        for boundary,rows in value['rows'].items():
            assert len(rows)==len(expected);clean=[];stage_sum=0
            for row,case in zip(rows,expected,strict=True):
                assert row['name']==case['name'] and row['dtype']==case['dtype']
                if mode=='numerics':
                    helper=row['helper_admitted']
                    assert helper==(case['eligible'] if role=='candidate' and case['dtype']=='Float' else None)
                    admitted+=helper is True
                if boundary=='provider' and case['layout'].startswith('null-'):
                    parameter={'null-condition':'condition','null-x':'X','null-y':'Y'}[case['layout']]
                    assert row['failure']==f'The required input parameter {parameter} is missing or null.' and row['inputs']
                    assert set(row)=={'name','dtype','failure','inputs','helper_admitted'}
                elif case.get('error'):
                    assert row['error']==case['error'] and row['inputs']
                    assert set(row)=={'name','dtype','error','inputs','helper_admitted'}
                else:
                    shape=list(np.broadcast_shapes(tuple(case['cshape']),tuple(case['xshape']),tuple(case['yshape'])))
                    assert row['shape']==shape and row['values']==int(np.prod(shape))
                    assert re.fullmatch('[0-9a-f]{64}',row['output'])
                    wanted_stage=0 if boundary=='tensor' else (-1 if mode=='codegen' else (2 if role=='candidate' and case['provider_attempt'] and not case['provider_eligible'] else 1))
                    assert row['validation_stages']==wanted_stage;stage_sum+=wanted_stage
                    if mode=='numerics':
                        assert all(row[k] for k in ['oracle','inputs','held','owned'])
                        assert re.fullmatch('[0-9a-f]{64}',row['mutated_output']);values+=row['values']
                        if case['name'] in prior_rows:
                            assert all(row[k]==prior_rows[case['name']][k] for k in ['shape','values','output','mutated_output'])
                    else:assert row['calls']==81
                    if 'raw_mask' in case or 'files' in case:
                        if 'raw_mask' in case:
                            c=np.array(case['raw_mask'],dtype=np.uint8)
                            x=np.array([bits[4]],dtype='<u4');y=np.resize(bits,int(np.prod(case['yshape']))).copy()
                        else:
                            c=np.fromfile(BASE/'bundle/fixtures'/case['files'][0],dtype=np.uint8).copy()
                            if case['mask']!='captured':
                                c[:]=0
                                if case['mask']=='true':c[:]=1
                                elif case['mask']=='first':c[0]=1
                                elif case['mask']=='last':c[-1]=1
                                else:assert case['mask']=='alternating';c[::2]=1
                            dtype='<u4' if case['dtype']=='Float' else '<u8'
                            x=np.fromfile(BASE/'bundle/fixtures'/case['files'][1],dtype=dtype).copy()
                            y=np.fromfile(BASE/'bundle/fixtures'/case['files'][2],dtype=dtype).copy()
                        def oracle():return np.where(c.astype(bool).reshape(case['cshape']),x.reshape(case['xshape']),y.reshape(case['yshape'])).tobytes()
                        assert hashlib.sha256(oracle()).hexdigest()==row['output']
                        if 'reference' in case:assert oracle()==(BASE/'bundle/fixtures'/case['reference']).read_bytes()
                        if mode=='numerics':
                            c[0]^=1;x[0]^=1;y[0]^=1
                            assert hashlib.sha256(oracle()).hexdigest()==row['mutated_output']
                clean.append({k:v for k,v in row.items() if k not in ['helper_admitted','validation_stages']})
            stages[name+'-'+boundary]=stage_sum;clean_boundaries[boundary]=clean
        assert [{k:v for k,v in row.items() if k!='failure'} for row in clean_boundaries['tensor'] if 'error' not in row]==[
            {k:v for k,v in row.items() if k!='failure'} for row in clean_boundaries['provider'] if 'error' not in row and 'failure' not in row]
        results[name]=clean_boundaries
        contract=value['contracts'];assert len(contract)==(20 if mode=='numerics' else 0)
        if mode=='numerics':
            names=['option-'+k for k in ['null','default','scalar','simd','intrinsics','memory','parallel2','no-pool']]+[
                'missing-condition','missing-X','missing-Y','wrong-condition','mismatched-operands','unsupported-int8','unsupported-int16','unsupported-uint16',
                'invalid-parallelism','intrinsics-without-simd','missing-before-options','options-before-condition-type']
            assert [r['name'] for r in contract]==names and all(r['inputs'] for r in contract)
            expected_hash=hashlib.sha256(np.resize(bits,4096).tobytes()).hexdigest()
            for row in contract[:8]:assert row['output']==expected_hash and row['values']==4096 and row['owned']
            for row in contract[8:16]+contract[18:19]:assert row['failure'] and isinstance(row['failure'],str)
            for i,kind,param in [(16,'System.ArgumentOutOfRangeException','MaxDegreeOfParallelism'),(17,'System.ArgumentException','UseIntrinsics'),(19,'System.ArgumentOutOfRangeException','MaxDegreeOfParallelism')]:
                assert contract[i]['error']==kind and contract[i]['parameter']==param
            contracts[name]=contract
        else:
            disasm=(folder/'logs'/(name+'.stdout')).read_text()
            for method in ['Lokad.Onnx.CPUExecutionProvider:Where','Lokad.Onnx.Tensor`1[float]:Where','Program:Invoke[float]']:
                assert 'Assembly listing for method '+method in disasm
            if role=='candidate':assert 'Assembly listing for method Lokad.Onnx.UniformScalarWhere:Try[float]' in disasm and '(FullOpts)' in disasm
    reference=results['current-numerics-256']
    assert all(results[f'{role}-numerics-{width}']==reference for role in ['current','candidate'] for width in [256,512])
    assert all(v==contracts['current-numerics-256'] for v in contracts.values())
    assert results['current-codegen-512']==results['candidate-codegen-512']
    assert admitted==292
    analysis=dict(passed=True,products=payload['products'],consumer=built['consumer'],numerical_workers=4,
        cases_per_boundary=203,boundaries=2,provider_contract_cases_per_worker=20,values_all_numerical_workers=values,
        candidate_helper_admissions_both_boundaries_modes=admitted,provider_fast_cases_per_census=35,
        codegen_workers=2,codegen_cases_per_boundary=188,calls_per_codegen_case=81,profile_validation_stage_sums=stages,
        independent_coordinate_oracle=True,captured_and_raw_numpy_bit_oracle=True,all_selected_bits_exact=True,
        original_131_case_hashes_exact=True,input_ownership_and_held_outputs=True,resources=resources,peak_rss=peak,
        root_product_changed=False,no_performance_measurement=True)
    save(BASE/'analysis.json',analysis)
    save(BASE/'closed.json',dict(passed=True,files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()},analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),**analysis)))


if __name__=='__main__':main()
