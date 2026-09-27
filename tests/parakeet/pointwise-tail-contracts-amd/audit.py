"""Audit compiled scope and every raw/fallback contract without scoring time."""
import base64
import json
from pathlib import Path
import sys
from run import BASE,PRELUDE,pin,read,write,ssh,prepared


def collected(kind):
    prepared();folder=BASE/(kind+'-collected');spec=read(BASE/'bundle/spec.json')
    transfer=read(BASE/(kind+'-transfer.json'));receipt=read(folder/(kind+'-collection.json'))
    assert transfer['passed'] and transfer['archive']==pin(BASE/(kind+'-results.tar.gz'))
    assert transfer['collection']==pin(folder/(kind+'-collection.json')) and receipt['terminal'] and receipt['code']==0
    for name,wanted in receipt['files'].items():assert pin(folder/name)==wanted,name
    assert pin(folder/'spec.json')==pin(BASE/'bundle/spec.json')
    for name,wanted in spec['files'].items():assert pin(folder/name)==wanted,name
    state=read(folder/(kind+'-state.json'))
    assert state['complete'] and state['code']==0 and receipt['state']==pin(folder/(kind+'-state.json'))
    assert state['supervisor']==read(BASE/(kind+'-deployment.json'))
    assert receipt['identities']==[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    names=['sdk-version','core-restore','core-build','contract-restore','contract-build','bridge-restore','bridge-build','inventory'] if kind=='build' else spec['modes']
    assert [r['name'] for r in state['runs']]==names
    limits=spec[kind+'_limits'];resources=[]
    for job in state['runs']:
        assert job['complete'] and job['code']==0 and job['seconds']<limits['seconds']
        assert job['preflight']['available']>=limits['available_before'] and job['preflight']['tmpfs']>=limits['tmpfs_before']
        samples=[json.loads(s) for s in (folder/'logs'/(job['name']+'.resources.jsonl')).read_text().splitlines()]
        assert len(samples)==job['samples']>0
        for sample in samples:
            assert sample['seconds']<limits['seconds'] and sample['rss']<limits['rss']
            assert min(sample['available'],sample['tmpfs'])>=spec['minimum_free'] and sample['output']<spec['output_limit']
            assert sample['rss']==sum(m['rss'] for m in sample['members'])
            for member in sample['members']:
                assert job['members'][str(member['pid'])]==member['birth'] and member['affinity']==[2]
                assert member['threads'] and all(t==[2] for t in member['threads'])
        gaps=[samples[0]['seconds']]+[b['seconds']-a['seconds'] for a,b in zip(samples,samples[1:])]+[job['seconds']-samples[-1]['seconds']]
        assert all(0<=g<10 for g in gaps)
        resources.append(dict(name=job['name'],samples=len(samples),seconds=job['seconds'],peak_rss=max(s['rss'] for s in samples)))
    built=read(folder/'built.json');assert built['passed'] and built['inventory']==pin(folder/'logs/instructions.json')
    for name,wanted in built['runtime'].items():assert pin(folder/'runtime'/name)==wanted,name
    for name,wanted in spec['runtime'].items():
        assert built['runtime']['baseline/'+name]==wanted,name
        if name!='Lokad.Onnx.dll':assert built['runtime']['candidate/'+name]==wanted,name
    for suffix in ['dll','deps.json','runtimeconfig.json']:
        assert built['runtime']['baseline/TailContracts.'+suffix]==built['runtime']['candidate/TailContracts.'+suffix]
    assert len(built['runtime'])==2*(len(spec['runtime'])+3)
    for role in ['baseline','candidate']:
        assert built['products'][role]=={n:built['runtime'][role+'/'+n] for n in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']}
    return folder,spec,receipt,state,built,resources


def build():
    assert not (BASE/'build-review.json').exists()
    folder,spec,receipt,state,built,resources=collected('build')
    inventory=read(folder/'logs/instructions.json');assert inventory['inventory_complete']
    assert [r['assembly'] for r in inventory['observations']]==['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']
    assert inventory['release']['sha256']==spec['runtime']['Lokad.Onnx.dll']['sha256']
    scope=[]
    for row in inventory['observations']:
        core=row['assembly']=='Lokad.Onnx.dll'
        assert row['before_sha256']==built['products']['baseline'][row['assembly']]['sha256']
        assert row['after_sha256']==built['products']['candidate'][row['assembly']]['sha256']
        assert row['methods']==(3286 if core else 697) and not row['removed']
        if core:
            assert len(row['differences'])==1 and {n.split('::')[1] for n in row['differences']}==set(spec['changed_methods'])
            assert len(row['added'])==2 and {n.split('::')[1] for n in row['added']}==set(spec['added_methods'])
            assert all(n.startswith('Lokad.Onnx.MathOps::') for n in row['differences']+row['added'])
        else:assert not row['differences'] and not row['added']
        assert row['unchanged_methods']==row['methods']-(1 if core else 0)
        assert row['public_surface_equal'] and row['public_surface']==row['public_surface_after']
        assert row['assembly_attributes_before']==row['assembly_attributes_after']
        assert all(row['method_flags_after'][n]==v for n,v in row['method_flags_before'].items())
        assert set(row['candidate_methods'])==set(row['differences']+row['added'])
        scope.append(dict(assembly=row['assembly'],unchanged=row['unchanged_methods'],changed=row['differences'],added=row['added']))
    warnings=[]
    for job in state['runs']:
        text=(folder/'logs'/(job['name']+'.stdout')).read_text()+(folder/'logs'/(job['name']+'.stderr')).read_text()
        assert ': error ' not in text
        warnings.extend(s for s in text.splitlines() if ': warning ' in s)
    assert len(warnings)==4 and all('Zzz.WideProjectionEntry.cs(20,' in s and 'warning CS8604:' in s for s in warnings),warnings
    result=dict(passed=True,built=pin(folder/'built.json'),products=built['products'],scope=scope,warnings=warnings,
        resources=resources,source=spec['source'],release_admitted=False,no_model_execution=True,reviewer=pin(Path(__file__)))
    write(BASE/'build-review.json',result)
    encoded=base64.b64encode((BASE/'build-review.json').read_bytes()).decode()
    transferred=ssh(PRELUDE+f'''
from remote import verify,read,live,pin
import base64
verify();state=read(base/'build-state.json')
assert state['complete'] and state['code']==0 and not live(state['supervisor'])
assert all(not live(dict(pid=int(p),birth=b)) for r in state['runs'] for p,b in r['members'].items())
assert pin(base/'built.json')=={result['built']!r}
with (base/'build-review.json').open('xb') as f:f.write(base64.b64decode({encoded!r}))
print(json.dumps(dict(passed=True,review=pin(base/'build-review.json'))))
''')
    assert transferred['review']==pin(BASE/'build-review.json');write(BASE/'build-review-transferred.json',transferred)
    print(json.dumps(dict(**transferred,scope=scope)))


def validate_cases(result,spec,raw):
    assert result['completed'] and len(result['results'])==(len(spec['raw_cases']) if raw else len(spec['fallback_cases'])*3)
    failures=[]
    if raw:
        for wanted,row in zip(spec['raw_cases'],result['results'],strict=True):
            assert {k:row[k] for k in wanted}==wanted
            if not row['bit_exact']:failures.append(row);continue
            assert row['inputs_immutable'] and row['guards_intact'] and row['allocated_bytes']==0
            assert len(row['output_sha256'])==64
    else:
        for wanted,row in zip([(m,n,k,option) for m,n,k in spec['fallback_cases'] for option in ['auto','simd','scalar']],result['results'],strict=True):
            assert (row['m'],row['n'],row['k'],row['option'])==wanted
            assert row['bit_exact'] and row['inputs_immutable'] and len(row['output_sha256'])==64
    assert result['failed']==len(failures) and result['passed']==(not failures)
    return failures


def capture():
    assert not (BASE/'closed.json').exists()
    folder,spec,receipt,state,built,resources=collected('capture')
    assert pin(folder/'build-review.json')==pin(BASE/'build-review.json')
    assert read(BASE/'build-collected/build-state.json')['ended']<state['started']
    results={};failures={}
    for job in state['runs']:
        mode=job['name'];result=read(folder/'probe'/mode/'result.json');results[mode]=result
        assert result['mode']==mode and result['pid']==job['owner']['pid'] and result['runtime']=='.NET 10.0.8'
        role='baseline' if mode=='scalar-baseline' else 'candidate'
        assert result['core_sha256']==built['products'][role]['Lokad.Onnx.dll']['sha256']
        assert result['consumer_sha256']==built['runtime'][role+'/TailContracts.dll']['sha256']
        flags={'DOTNET_JitDisasm':spec['disasm']} if mode=='normal' else ({'DOTNET_EnableAVX512':'0'} if mode=='avx512-disabled' else {'DOTNET_EnableHWIntrinsic':'0'})
        assert result['flags']==flags
        failures[mode]=validate_cases(result,spec,not mode.startswith('scalar-'))
    assert results['scalar-baseline']['results']==results['scalar-candidate']['results']
    passed=not any(failures.values())
    if passed:assert results['normal']['results']==results['avx512-disabled']['results']
    analysis=dict(passed=passed,completed=True,products=built['products'],resources=resources,
        cases={mode:len(value['results']) for mode,value in results.items()},failures=failures,
        normal_disabled_exact=passed,scalar_baseline_candidate_exact=True,source=spec['source'],
        codegen_review_pending=True,no_performance_measurement=True,release_admitted=False,
        reviewer=pin(Path(__file__)),disassembly=pin(folder/'logs/normal.stdout'))
    write(BASE/'analysis.json',analysis)
    write(BASE/'closed.json',dict(passed=passed,completed=True,analysis=pin(BASE/'analysis.json'),
        terminal_owners=receipt['identities'],files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),passed=passed,cases=analysis['cases'],failures={k:len(v) for k,v in failures.items()},resources=resources)))


if __name__=='__main__':{'build':build,'capture':capture}[sys.argv[1]]()
