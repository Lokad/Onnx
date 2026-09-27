"""Retain exact non-NaN bits, classification, ownership and payload telemetry."""
import json
from pathlib import Path
import sys
from run import BASE,pin,read,write,prepared


def collected(kind):
    prepared();folder=BASE/(kind+'-collected');receipt=read(folder/(kind+'-collection.json'))
    assert receipt['terminal'] and receipt['code']==0
    transfer=read(BASE/(kind+'-transfer.json'))
    assert transfer['passed'] and transfer['archive']==pin(BASE/(kind+'-results.tar.gz'))
    assert transfer['collection']==pin(folder/(kind+'-collection.json'))
    for name,wanted in receipt['files'].items():assert pin(folder/name)==wanted,name
    assert pin(folder/'spec.json')==pin(BASE/'bundle/spec.json')
    spec=read(folder/'spec.json');state=read(folder/(kind+'-state.json'));built=read(folder/'built.json')
    assert state['complete'] and state['code']==0 and state['supervisor']==read(BASE/(kind+'-deployment.json'))
    names=['sdk-version','contract-restore','contract-build'] if kind=='build' else [j['name'] for j in spec['jobs']]
    assert [r['name'] for r in state['runs']]==names
    resources=[]
    for job in state['runs']:
        assert job['complete'] and job['code']==0 and job['seconds']<spec[kind+'_limits']['seconds']
        samples=[json.loads(line) for line in (folder/'logs'/(job['name']+'.resources.jsonl')).read_text().splitlines()]
        assert len(samples)==job['samples']>0
        for s in samples:
            assert s['rss']<spec[kind+'_limits']['rss'] and s['seconds']<spec[kind+'_limits']['seconds']
            assert min(s['available'],s['tmpfs'])>=spec['minimum_free'] and s['output']<spec['output_limit']
            assert s['rss']==sum(m['rss'] for m in s['members'])
            for m in s['members']:
                assert job['members'][str(m['pid'])]==m['birth'] and m['affinity']==[2] and m['threads'] and all(t==[2] for t in m['threads'])
        gaps=[samples[0]['seconds']]+[b['seconds']-a['seconds'] for a,b in zip(samples,samples[1:])]+[job['seconds']-samples[-1]['seconds']]
        assert all(0<=g<10 for g in gaps)
        resources.append(dict(name=job['name'],seconds=job['seconds'],samples=len(samples),peak_rss=max(s['rss'] for s in samples)))
    assert built['products']==spec['products']
    for name,wanted in built['runtime'].items():assert pin(folder/'runtime'/name)==wanted,name
    for role,files in spec['runtimes'].items():
        for name,wanted in files.items():assert built['runtime'][role+'/'+name]==wanted,name
    for suffix in ['dll','deps.json','runtimeconfig.json']:assert built['runtime']['baseline/TailContracts.'+suffix]==built['runtime']['candidate/TailContracts.'+suffix]
    assert read(folder/'built-baseline.json')['products']==dict(baseline=spec['products']['baseline'],candidate=spec['products']['baseline'])
    return folder,spec,state,built,resources


def build():
    folder,spec,state,built,resources=collected('build')
    for job in state['runs']:
        log=(folder/'logs'/(job['name']+'.stdout')).read_text()+(folder/'logs'/(job['name']+'.stderr')).read_text()
        assert ': warning ' not in log and ': error ' not in log
    value=dict(passed=True,built=pin(folder/'built.json'),products=built['products'],resources=resources,no_product_build=True,compiled_scope=spec['compiled_scope'],reviewer=pin(Path(__file__)))
    write(BASE/'build-review.json',value);print(json.dumps(dict(review=pin(BASE/'build-review.json'),products=value['products'])))


def validate_raw(result,spec):
    assert result['completed'] and result['comparison_selftests']==9
    assert len(result['results'])==len(spec['raw_cases'])
    failures=[];payload_cases=[]
    for wanted,row in zip(spec['raw_cases'],result['results'],strict=True):
        assert {k:row[k] for k in wanted}==wanted
        if not row['arithmetic_contract_passed']:failures.append(row);continue
        assert row['non_nan_bits_exact'] and row['nan_classification_exact']
        assert row['inputs_immutable'] and row['guards_intact'] and row['allocated_bytes']==0
        count=row['nan_payload_differences'];assert type(count) is int and count>=0 and row['bit_exact']==(count==0)
        assert len(row['output_sha256'])==len(row['baseline_output_sha256'])==64
        if row['bit_exact']:assert row['output_sha256']==row['baseline_output_sha256']
        if count:
            assert row['exceptional'];payload_cases.append(dict(wanted,differences=count))
    assert result['failed']==len(failures) and result['passed']==(not failures)
    return failures,payload_cases


def capture():
    folder,spec,state,built,resources=collected('capture')
    assert read(BASE/'build-review.json')['passed']
    results={};failures={};payloads={}
    for declared,job in zip(spec['jobs'],state['runs'],strict=True):
        name=declared['name'];mode=declared['mode'];role=declared['role'];result=read(folder/'probe'/name/'result.json');results[name]=result
        assert result['mode']==mode and result['pid']==job['owner']['pid'] and result['runtime']=='.NET 10.0.8'
        assert result['core_sha256']==spec['products'][role]['Lokad.Onnx.dll']['sha256']
        assert result['consumer_sha256']==built['runtime'][role+'/TailContracts.dll']['sha256']
        flags={'DOTNET_EnableHWIntrinsic':'0'} if mode.startswith('scalar-') else {'DOTNET_JitDisasm':spec['disasm']}
        if mode=='avx512-disabled':flags['DOTNET_EnableAVX512']='0'
        assert result['flags']==flags
        if mode.startswith('scalar-'):
            assert result['completed'] and result['passed'] and result['failed']==0 and result['comparison_selftests']==9
            assert len(result['results'])==len(spec['fallback_cases'])*3
            for wanted,r in zip([(m,n,k,o) for m,n,k in spec['fallback_cases'] for o in ['auto','simd','scalar']],result['results'],strict=True):
                assert tuple(r[k] for k in ['m','n','k','option'])==wanted and r['bit_exact'] and r['inputs_immutable']
            failures[name]=[];payloads[name]=[]
        else:failures[name],payloads[name]=validate_raw(result,spec)
    assert results['scalar-baseline']['results']==results['scalar-candidate']['results']
    finite_sets=[]
    for name in ['baseline-normal','baseline-avx512-disabled','candidate-normal','candidate-avx512-disabled']:
        finite_sets.append([r for r in results[name]['results'] if not r['exceptional']])
    assert all(s==finite_sets[0] for s in finite_sets[1:])
    passed=not any(failures.values())
    value=dict(arithmetic_contract_passed=passed,all_bits_exact=passed and not any(payloads.values()),failures=failures,nan_payload_cases=payloads,
        cases={n:len(r['results']) for n,r in results.items()},products=built['products'],resources=resources,
        baseline_diagnosis=spec['baseline_diagnosis'],original_failed_closure=spec['original_failed_closure'],compiled_scope=spec['compiled_scope'],
        finite_cross_mode_exact=True,scalar_results_exact=True,codegen_review_pending=True,no_performance_measurement=True,release_admitted=False,reviewer=pin(Path(__file__)))
    write(BASE/'analysis.json',value)
    write(BASE/'closed.json',dict(arithmetic_contract_passed=passed,completed=True,analysis=pin(BASE/'analysis.json'),files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),arithmetic_contract_passed=passed,all_bits_exact=value['all_bits_exact'],cases=value['cases'],failures={n:len(f) for n,f in failures.items()},payload_cases={n:len(p) for n,p in payloads.items()})))


if __name__=='__main__':{'build':build,'capture':capture}[sys.argv[1]]()
