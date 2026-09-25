"""Audit exact binaries, every planned case and resources; retain failed verdicts."""
import base64
import json
from pathlib import Path
import sys
from run import BASE, ROOT, TOOLS, PRELUDE, pin, read, write, ssh, prepared, cases


def collected(kind):
    prepared(); folder=BASE/(kind+'-collected');spec=read(BASE/'bundle/spec.json')
    receipt=read(folder/(kind+'-collection.json'));transfer=read(BASE/(kind+'-transfer.json'))
    assert transfer['passed'] and transfer['archive']==pin(BASE/(kind+'-results.tar.gz'))
    assert transfer['collection']==pin(folder/(kind+'-collection.json'))
    assert receipt['terminal'] and receipt['code']==0
    for name,wanted in receipt['files'].items():assert pin(folder/name)==wanted,name
    assert pin(folder/'spec.json')==pin(BASE/'bundle/spec.json')
    for name,wanted in spec['files'].items():assert pin(folder/name)==wanted,name
    state=read(folder/(kind+'-state.json'))
    assert state['complete'] and state['code']==0 and receipt['state']==pin(folder/(kind+'-state.json'))
    assert state['supervisor']==read(BASE/(kind+'-deployment.json'))
    assert receipt['identities']==[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    expected=['sdk-version','consumer-restore','consumer-build'] if kind=='build' else ['native','hardware-disabled']
    assert [r['name'] for r in state['runs']]==expected
    resources=[];limits=spec[kind+'_limits']
    for run in state['runs']:
        assert run['complete'] and run['code']==0 and run['seconds']<limits['seconds']
        assert run['preflight']['available']>=limits['available_before'] and run['preflight']['tmpfs']>=limits['tmpfs_before']
        samples=[json.loads(s) for s in (folder/'logs'/(run['name']+'.resources.jsonl')).read_text().splitlines()]
        assert len(samples)==run['samples']>0
        for sample in samples:
            assert sample['seconds']<limits['seconds'] and sample['rss']<limits['rss']
            assert min(sample['available'],sample['tmpfs'])>=spec['minimum_free'] and sample['output']<spec['output_limit']
            assert sample['rss']==sum(m['rss'] for m in sample['members'])
            for member in sample['members']:
                assert run['members'][str(member['pid'])]==member['birth'] and member['affinity']==[2]
                assert member['threads'] and all(t==[2] for t in member['threads'])
        gaps=[samples[0]['seconds']]+[b['seconds']-a['seconds'] for a,b in zip(samples,samples[1:])]+[run['seconds']-samples[-1]['seconds']]
        assert all(0<=g<10 for g in gaps)
        resources.append(dict(name=run['name'],samples=len(samples),seconds=run['seconds'],peak_rss=max(s['rss'] for s in samples)))
    built=read(folder/'built.json');assert built['passed'] and not built['product_rebuilt']
    for name,wanted in built['runtime'].items():assert pin(folder/'runtime'/name)==wanted
    for name,wanted in spec['runtime_files'].items():assert built['runtime'][name]==wanted
    assert set(built['runtime'])==set(spec['runtime_files'])|{'PackedLogicalWeightProbe.dll','PackedLogicalWeightProbe.deps.json','PackedLogicalWeightProbe.runtimeconfig.json'}
    return folder,spec,receipt,state,built,resources


def build():
    assert not (BASE/'build-review.json').exists()
    folder,spec,receipt,state,built,resources=collected('build')
    assert (folder/'logs/sdk-version.stdout').read_text().strip()=='10.0.204'
    for name in ['consumer-restore','consumer-build']:
        output=(folder/'logs'/(name+'.stdout')).read_text()+(folder/'logs'/(name+'.stderr')).read_text()
        assert ': warning ' not in output and ': error ' not in output
    output=(folder/'logs/consumer-build.stdout').read_text()
    assert '0 Warning(s)' in output and '0 Error(s)' in output
    for name in ['Program','PackedLogicalWeight']:
        assert pin(folder/'source'/(name+'.cs'))==pin(TOOLS/(name+'.cs.txt'))
    result=dict(passed=True,product_rebuilt=False,zero_added_warnings=True,built=pin(folder/'built.json'),
        collection=pin(folder/'build-collection.json'),transfer=pin(BASE/'build-transfer.json'),
        sources={n:pin(folder/'source'/(n+'.cs')) for n in ['Program','PackedLogicalWeight']},
        product=spec['product'],resources=resources,release_admitted=False,reviewer=pin(Path(__file__)))
    write(BASE/'build-review.json',result)
    encoded=base64.b64encode((BASE/'build-review.json').read_bytes()).decode()
    transferred=ssh(PRELUDE+f'''
from remote import verify,live,read,pin
import base64
verify();state=read(base/'build-state.json')
assert state['complete'] and state['code']==0 and not live(state['supervisor'])
assert all(not live(dict(pid=int(p),birth=b)) for r in state['runs'] for p,b in r['members'].items())
assert pin(base/'built.json')=={result['built']!r}
with (base/'build-review.json').open('xb') as f:f.write(base64.b64decode({encoded!r}))
print(json.dumps(dict(passed=True,review=pin(base/'build-review.json'))))
''')
    assert transferred['review']==pin(BASE/'build-review.json')
    write(BASE/'build-review-transferred.json',transferred)
    print(json.dumps(dict(passed=True,review=transferred['review'],resources=resources)))


def capture():
    assert not (BASE/'closed.json').exists()
    folder,spec,receipt,state,built,resources=collected('capture')
    assert pin(folder/'build-review.json')==pin(BASE/'build-review.json')
    assert read(BASE/'build-review.json')['built']==pin(folder/'built.json')
    assert read(BASE/'build-collected/build-state.json')['ended']<state['started']
    verdicts={}; failures=[]
    for mode in ['native','hardware-disabled']:
        result=read(folder/'probe'/mode/'result.json')
        assert result['completed'] and result['mode']==mode and result['runtime']=='.NET 10.0.8'
        assert result['affinity']==4 and result['processor_count']==1
        for name in ['diagnostic_only','no_model_execution','no_application_score','no_product_change']:assert result[name]
        assert result['core_sha256']==spec['product']['Lokad.Onnx.dll']['sha256']
        assert result['runner_sha256']==built['runtime']['PackedLogicalWeightProbe.dll']['sha256']
        assert result['spec_sha256']==pin(folder/'spec.json')['sha256']
        assert result['flags']==({} if mode=='native' else {'DOTNET_EnableHWIntrinsic':'0'})
        assert all(result[name]==(mode=='native') for name in ['fma','avx2','avx512'])
        assert [r['name'] for r in result['kernels']]==['ShortWidePackPanelsB','ShortWideMultiply2Rows','ShortWideMultiply3Rows','ShortWideMultiplyRemainder']
        assert all(k['assembly']=='Lokad.Onnx' and k['token']>0 for k in result['kernels'])
        assert len({k['module'] for k in result['kernels']})==1
        expected=cases(mode);assert expected==spec['cases'][mode]
        assert [r['id'] for r in result['cases']]==[r['id'] for r in expected]
        for row,case in zip(result['cases'],expected):
            assert read(folder/'probe'/mode/(row['id']+'.json'))==row
            assert type(row['passed']) is bool
            if not row['passed']:
                assert row['error'];failures.append(dict(mode=mode,**row));continue
            evidence=row['evidence'];kind=case['kind']
            for axis in ['n','k','m','mode']:
                if axis in case:assert evidence[axis]==case[axis]
            if kind in ['layout','packed','fallback']:
                assert evidence['bit_exact'] and len(evidence['source_sha256'])==64 and len(evidence['packed_sha256'])==64
            if kind=='layout':
                assert evidence['elements']==case['n']*case['k'] and evidence['independent_copies']
                assert evidence['views']==(case['n']<10)
            elif kind=='bounds':assert evidence['checks']==8 and evidence['raw_storage_not_exposed']
            elif kind=='fallback':
                assert evidence['DenseMaterializations']==1
                assert evidence['DenseBytes']==evidence['copy_bytes']==case['n']*case['k']*4
                assert len(evidence['output_sha256'])==64
            elif kind=='packed':
                odd=case['m']==167
                assert evidence['rows']==(166 if odd else 225) and evidence['remainder']==odd
                assert evidence['multiply']==('ShortWideMultiply2Rows' if odd else 'ShortWideMultiply3Rows')
                assert evidence['packed_reconstruction_bytes']==(16777216 if odd else 0)
                assert evidence['fallback_reconstruction_bytes']==16777216 and len(evidence['output_sha256'])==64
            elif kind=='ownership':
                assert not evidence['source_alive'] and not evidence['temporary_alive']
                assert evidence['retained_payload_bytes']==16777216 and evidence['no_dense_cache'] and evidence['forced_gc_unscored']
            elif kind=='unavailable':assert evidence['rejected']
        count=sum(not r['passed'] for r in result['cases'])
        assert result['failed']==count and result['passed']==(count==0)
        verdicts[mode]=result
    passed=not failures
    analysis=dict(passed=True,representation_passed=passed,diagnostic_only=True,release_admitted=False,
        no_product_change=True,no_application_score=True,no_model_execution=True,failures=failures,
        cases=sum(len(v['cases']) for v in verdicts.values()),modes=verdicts,resources=resources,
        existing_odd_remainder_requires_full_reconstruction=True,
        graph_dispatch_alias_and_invalidation_not_tested=True,product=spec['product'],
        failed_release_controls=spec['failed_release_controls'],ownership_closure=spec['ownership_closure'])
    write(BASE/'analysis.json',analysis)
    write(BASE/'closed.json',dict(passed=True,representation_passed=passed,analysis=pin(BASE/'analysis.json'),
        collection=pin(folder/'capture-collection.json'),transfer=pin(BASE/'capture-transfer.json'),
        terminal_owners=receipt['identities'],reviewer=pin(Path(__file__)),
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),representation_passed=passed,cases=analysis['cases'],failures=failures,resources=resources)))


if __name__=='__main__':{'build':build,'capture':capture}[sys.argv[1]]()
