"""Independently reconcile runtime identities, actual corpus shapes and traffic."""
import base64
import json
from pathlib import Path
import sys
from run import BASE,ROOT,TOOLS,PRELUDE,JOBS,pin,read,write,ssh,prepared


def collected(kind):
    prepared();folder=BASE/(kind+'-collected');spec=read(BASE/'bundle/spec.json')
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
    expected=['sdk-version','consumer-restore','consumer-build'] if kind=='build' else JOBS
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
    assert set(built['runtime'])=={role+'/'+n for role,files in spec['runtime_files'].items() for n in [*files,'OwnedWeightCounters.dll','OwnedWeightCounters.deps.json','OwnedWeightCounters.runtimeconfig.json']}
    for name,wanted in built['runtime'].items():assert pin(folder/'runtime'/name)==wanted,name
    for role,files in spec['runtime_files'].items():
        for name,wanted in files.items():assert built['runtime'][role+'/'+name]==wanted
    for suffix in ['dll','deps.json','runtimeconfig.json']:
        assert built['runtime']['selected/OwnedWeightCounters.'+suffix]==built['runtime']['candidate/OwnedWeightCounters.'+suffix]
    return folder,spec,receipt,state,built,resources


def build():
    assert not (BASE/'build-review.json').exists()
    folder,spec,receipt,state,built,resources=collected('build')
    assert (folder/'logs/sdk-version.stdout').read_text().strip()=='10.0.204'
    for name in ['consumer-restore','consumer-build']:
        text=(folder/'logs'/(name+'.stdout')).read_text()+(folder/'logs'/(name+'.stderr')).read_text()
        assert ': warning ' not in text and ': error ' not in text
    assert '0 Warning(s)' in text and '0 Error(s)' in text
    source=(folder/'source/Program.cs').read_text()
    assert 'GC.Collect(' not in source and 'new ParakeetTranscriber(' in source and 'OnnxImport.Load' not in source
    assert 'Profiler.BeginExecution(true)' in source and 'enc.LastCopyBytes' in source and 'enc.LastScratchBytes' in source
    assert 's.Time' not in source and 'owner.Transcribe(' not in source
    project=(folder/'source/OwnedWeightCounters.csproj').read_text();assert '<ProjectReference' not in project and project.count('<Reference ')==6
    result=dict(passed=True,product_rebuilt=False,zero_added_warnings=True,built=pin(folder/'built.json'),
        collection=pin(folder/'build-collection.json'),transfer=pin(BASE/'build-transfer.json'),source=pin(folder/'source/Program.cs'),
        resources=resources,products=spec['products'],model_closure=spec['model_closure'],census_closure=spec['census_closure'],
        release_admitted=False,reviewer=pin(Path(__file__)))
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
    assert pin(folder/'build-review.json')==pin(BASE/'build-review.json') and read(BASE/'build-review.json')['built']==pin(folder/'built.json')
    assert read(BASE/'build-collected/build-state.json')['ended']<state['started']
    weights={r['Name']:r for r in spec['weights']};assert len(weights)==96 and sum(not w['Cached'] for w in weights.values())==87
    jobs={}
    for name in JOBS:
        role,mode=name.split('-');path=folder/'probe'/name;result=read(path/'result.json')
        assert result['passed'] and result['role']==role and result['mode']==mode and result['runtime']=='.NET 10.0.8'
        assert result['affinity']==4 and result['processor_count']==1 and result['avx512']==(mode=='512')
        assert result['core_sha256']==spec['products'][role]['Lokad.Onnx.dll']['sha256']
        assert result['data_sha256']==spec['products'][role]['Lokad.Onnx.Data.dll']['sha256']
        assert result['runner_sha256']==built['runtime'][role+'/OwnedWeightCounters.dll']['sha256']
        assert result['spec_sha256']==pin(folder/'spec.json')['sha256']
        assert result['immutable_inputs'] and result['held_outputs_independent']
        assert not result['product_rebuilt'] and not result['application_scored'] and not result['forced_gc']
        assert len(result['records'])==20 and {p.name for p in path.iterdir()}=={'result.json'}|{f'{i:03}.json' for i in range(20)}
        for i,(record,case) in enumerate(zip(result['records'],spec['cases'],strict=True)):
            assert record==read(path/f'{i:03}.json') and record['passed']
            assert record['name']==case['name'] and record['input_hash']==case['raw_sha256']
            frames=case['expected']['encoded_frames'];assert record['frames']==frames>=48
            remainder=frames%2!=0 and frames%3!=0;assert record['remainder']==remainder
            assert type(record['copy_bytes']) is int and record['copy_bytes']>=0
            assert type(record['scratch_bytes']) is int and record['scratch_bytes']>=0
            assert set(record['frontend'])=={'features','features_lens'} and set(record['encoder'])=={'outputs','encoded_lengths'}
            assert all(len(v)==64 and set(v)<=set('0123456789abcdef') for v in [*record['frontend'].values(),*record['encoder'].values()])
            calls=record['calls'];assert len(calls)==96 and len({c['node'] for c in calls})==96 and {c['weight'] for c in calls}==set(weights)
            for call in calls:
                weight=weights[call['weight']];assert '/feed_forward' in call['node']
                assert call['b']==weight['Shape'] and call['a'] in [[frames,call['b'][0]],[1,frames,call['b'][0]]]
                owned=role=='candidate' and not weight['Cached'];assert call['owned']==owned
                assert call['copy_y_stages']==int(owned and remainder)
        jobs[name]=result
    comparisons=[];bytes_per_request=sum(w['Bytes'] for w in weights.values() if not w['Cached'])
    assert bytes_per_request==87*16777216==1459617792
    for mode in ['512','256']:
        rows=[]
        for a,b in zip(jobs['selected-'+mode]['records'],jobs['candidate-'+mode]['records'],strict=True):
            assert all(a[k]==b[k] for k in ['name','frames','remainder','input_hash','frontend','encoder'])
            assert [{k:c[k] for k in ['node','weight','a','b']} for c in a['calls']]==[{k:c[k] for k in ['node','weight','a','b']} for c in b['calls']]
            scratch=a['scratch_bytes']-b['scratch_bytes'];copies=b['copy_bytes']-a['copy_bytes']
            assert scratch==bytes_per_request,(mode,a['name'],'scratch',scratch)
            assert copies==(bytes_per_request if a['remainder'] else 0),(mode,a['name'],'copies',copies)
            rows.append(dict(name=a['name'],frames=a['frames'],scratch_reduction=scratch,copy_increase=copies,
                selected_copy=a['copy_bytes'],candidate_copy=b['copy_bytes'],selected_scratch=a['scratch_bytes'],candidate_scratch=b['scratch_bytes'],
                avoided_packs=87,reconstructions=sum(c['copy_y_stages'] for c in b['calls']),outputs_exact=True))
        comparison=dict(mode=mode,clips=rows,avoided_packs=sum(r['avoided_packs'] for r in rows),reconstructions=sum(r['reconstructions'] for r in rows),
            scratch_reduction=sum(r['scratch_reduction'] for r in rows),copy_increase=sum(r['copy_increase'] for r in rows))
        assert comparison['avoided_packs']==spec['prediction']['packs_per_corpus']==1740
        assert comparison['reconstructions']==spec['prediction']['reconstructions_per_corpus']==609
        for key in ['scratch_reduction','copy_increase']:assert comparison[key]==spec['prediction'][key]
        comparisons.append(comparison)
    analysis=dict(passed=True,products=spec['products'],model_closure=spec['model_closure'],census_closure=spec['census_closure'],
        build_review=pin(BASE/'build-review.json'),comparisons=comparisons,resources=resources,
        application_scored=False,release_admitted=False,failed_release_controls=spec['failed_release_controls'])
    write(BASE/'analysis.json',analysis)
    write(BASE/'closed.json',dict(passed=True,analysis=pin(BASE/'analysis.json'),collection=pin(folder/'capture-collection.json'),
        transfer=pin(BASE/'capture-transfer.json'),terminal_owners=receipt['identities'],reviewer=pin(Path(__file__)),
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),comparisons=[{k:v for k,v in c.items() if k!='clips'} for c in comparisons],resources=resources)))


if __name__=='__main__':{'build':build,'capture':capture}[sys.argv[1]]()
