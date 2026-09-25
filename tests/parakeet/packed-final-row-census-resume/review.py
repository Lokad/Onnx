"""Review the unchanged consumer binding and actual weight census."""
import base64
import json
from pathlib import Path
import sys
from run import BASE,ROOT,TOOLS,PRELUDE,pin,read,write,ssh,prepared,references,JOBS


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
    expected=[] if kind=='build' else JOBS
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
    assert set(built['runtime'])==set(spec['runtime_files'])|{'OwnedWeightCensus.dll','OwnedWeightCensus.deps.json','OwnedWeightCensus.runtimeconfig.json'}
    return folder,spec,receipt,state,built,resources


def build():
    assert not (BASE/'build-review.json').exists()
    folder,spec,receipt,state,built,resources=collected('build')
    assert not resources and built['binding_only'] and not built['consumer_rebuilt']
    assert all(built['runtime'][n]==v for n,v in spec['consumer_files'].items())
    source=(folder/'source/Program.cs').read_text()
    assert source.count('owner.Transcribe(')==1 and 'GC.Collect(' not in source
    assert source.index('Weights(graph, spec.GetProperty("weights"))')<source.index('owner.Transcribe(')
    assert 'RemoveBindings' not in source and 'RemoveMapEntries' not in source
    result=dict(passed=True,product_rebuilt=False,consumer_rebuilt=False,binding_only=True,consumer_build_review=spec['consumer_build_review'],built=pin(folder/'built.json'),
        collection=pin(folder/'build-collection.json'),transfer=pin(BASE/'build-transfer.json'),
        source=pin(folder/'source/Program.cs'),resources=resources,product=spec['product'],
        contracts=spec['contracts'],original_compiled_review=spec['compiled_review'],release_admitted=False,reviewer=pin(Path(__file__)))
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


def initial_worker(folder,spec):
    initial=spec['initial'];root=folder/'evidence/initial'
    for key in ['collection','state','spec','transfer','build_review']:
        name='build-review.json' if key=='build_review' else key+'.json'
        assert pin(root/name)==initial[key]
    transfer=read(root/'transfer.json');assert transfer['passed'] and transfer['archive']==initial['archive'] and transfer['collection']==initial['collection']
    receipt=read(root/'collection.json');state=read(root/'state.json');original=read(root/'spec.json')
    assert receipt['terminal'] and receipt['code']==state['code']==1 and state['complete'] and receipt['state']==initial['state']
    assert state['supervisor']==initial['supervisor'] and state['ended']<read(folder/'build-state.json')['started']
    assert [r['name'] for r in state['runs']]==['census-512','census-256']
    assert receipt['identities']==[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    run,missing=state['runs'];assert run['complete'] and run['code']==0
    assert not missing['complete'] and missing['code'] is None and missing['samples']==0 and not missing['members'] and 'owner' not in missing
    assert missing['preflight']['available']<original['capture_limits']['available_before']
    assert missing['preflight']['tmpfs']>=original['capture_limits']['tmpfs_before'] and 'preflight' in state['error']
    for key in ['product','runtime_files','consumer_files','case','weights','retained','capture_limits','minimum_free','output_limit','failed_release_controls']:
        assert spec[key]==original[key],key
    for path in (folder/'probe/512').iterdir():assert pin(path)==receipt['files']['probe/512/'+path.name]
    assert pin(folder/'probe/512/result.json')==initial['selected_result']
    limits=spec['capture_limits']
    assert run['seconds']<limits['seconds'] and run['preflight']['available']>=limits['available_before'] and run['preflight']['tmpfs']>=limits['tmpfs_before']
    log=root/'census-512.resources.jsonl';assert pin(log)==receipt['files']['logs/census-512.resources.jsonl']
    samples=[json.loads(s) for s in log.read_text().splitlines()];assert len(samples)==run['samples']>0
    for sample in samples:
        assert sample['seconds']<limits['seconds'] and sample['rss']<limits['rss']
        assert min(sample['available'],sample['tmpfs'])>=spec['minimum_free'] and sample['output']<spec['output_limit']
        assert sample['rss']==sum(m['rss'] for m in sample['members'])
        for member in sample['members']:
            assert run['members'][str(member['pid'])]==member['birth'] and member['affinity']==[2]
            assert member['threads'] and all(t==[2] for t in member['threads'])
    gaps=[samples[0]['seconds']]+[b['seconds']-a['seconds'] for a,b in zip(samples,samples[1:])]+[run['seconds']-samples[-1]['seconds']]
    assert all(0<=g<10 for g in gaps)
    for name in JOBS:
        observations=read(folder/(name+'-preflight-wait.json'));assert observations
        assert observations[-1]['available']>=limits['available_before']
        assert all(0<=r['seconds']<spec['preflight_wait_seconds'] and r['tmpfs']>=limits['tmpfs_before'] for r in observations)
        assert all(a['seconds']<=b['seconds'] for a,b in zip(observations,observations[1:]))
    return dict(name=run['name'],samples=len(samples),seconds=run['seconds'],peak_rss=max(s['rss'] for s in samples),retained=True)


def capture():
    assert not (BASE/'closed.json').exists()
    folder,spec,receipt,state,built,resources=collected('capture')
    assert pin(folder/'build-review.json')==pin(BASE/'build-review.json')
    assert read(BASE/'build-review.json')['built']==pin(folder/'built.json')
    assert read(BASE/'build-collected/build-state.json')['ended']<state['started']
    resources.insert(0,initial_worker(folder,spec))
    expected={r['Name']:r for r in spec['weights']}
    packed={r['Name']:r['PackedHash'] for r in spec['retained'] if r['PackedHash'] is not None}
    modes=[]
    for mode in ['512','256']:
        path=folder/'probe'/mode;result=read(path/'result.json')
        assert result['passed'] and result['mode']==mode and result['runtime']=='.NET 10.0.8'
        assert result['affinity']==4 and result['processor_count']==1 and result['fma']
        assert result['avx512']==(mode=='512')
        assert result['core_sha256']==spec['product']['Lokad.Onnx.dll']['sha256']
        assert result['data_sha256']==spec['product']['Lokad.Onnx.Data.dll']['sha256']
        assert result['runner_sha256']==built['runtime']['OwnedWeightCensus.dll']['sha256']
        assert result['spec_sha256']==(spec['initial']['spec']['sha256'] if mode=='512' else pin(folder/'spec.json')['sha256'])
        for key in ['public_request_passed','idempotent','identities_preserved','logical_hashes_exact','pcm_unchanged','contexts_share_weights']:
            assert result[key],key
        for key in ['forced_gc','product_rebuilt','application_scored']:assert not result[key],key
        assert result['owned_count']==87 and result['owned_bytes']==1459617792
        assert result['retained_maps']==37 and result['retained_clone_bytes']==268435456
        assert result['original_weight_count']==96 and result['original_dense_count']==9 and result['initializer_count']==649
        before=read(path/'before.json');after=read(path/'after.json')
        assert before['weights']==after['weights'] and before['maps']==after['maps']
        assert len(before['weights'])==96 and {r['Name'] for r in before['weights']}==set(expected)
        for row in before['weights']:
            wanted=expected[row['Name']]
            assert row['Shape']==wanted['Shape'] and row['Cached']==wanted['Cached'] and row['LogicalHash']==wanted['Hash']
            assert row['Bytes']==wanted['Bytes']==16777216
            assert row['Kind']==('DenseTensor`1' if row['Cached'] else 'OwnedPackedTensor')
            assert len(row['PayloadHash'])==64
            if row['Cached']:assert row['PayloadHash']==row['LogicalHash']
        assert len(before['maps'])==37 and {r['Name']:r['Hash'] for r in before['maps']}==packed
        assert all(r['Name']=='packed:'+r['Source'] and r['Bytes']>0 for r in before['maps'])
        assert sum(r['Bytes'] for r in before['maps'])==268435456
        for snapshot in [before,after]:assert all(type(v) is int and v>=0 for v in snapshot['memory'].values())
        request=read(path/'request.json');actual=request['result'];wanted=spec['case']['expected']
        assert request['passed'] and request['name']==spec['case']['name'] and request['input_sha256']==spec['case']['raw_sha256']
        for left,right in dict(Text='text',TokenIds='token_ids',FrameIndices='frame_indices',DurationFrames='duration_frames',EncodedFrames='encoded_frames',DecoderCalls='decoder_calls').items():
            assert actual[left]==wanted[right],left
        assert wanted['stop_reason']=='EndOfAudio' and actual['StopReason']==0
        modes.append(dict(mode=mode,result=result,before=before,after=after,request=request))
    assert modes[0]['before']['weights']==modes[1]['before']['weights']
    assert modes[0]['before']['maps']==modes[1]['before']['maps']
    analysis=dict(passed=True,product=spec['product'],contracts=spec['contracts'],original_compiled_review=spec['compiled_review'],
        consumer_build_review=pin(BASE/'build-review.json'),modes=modes,resources=resources,
        release_admitted=False,application_scored=False,full_model_numerics_pending=True,full_public_corpus_pending=True,
        actual_reconstruction_census_pending=True,failed_release_controls=spec['failed_release_controls'],
        resumed_from=spec['initial'],original_failure_preserved=True,completed_mode_not_repeated=True)
    write(BASE/'analysis.json',analysis)
    write(BASE/'closed.json',dict(passed=True,analysis=pin(BASE/'analysis.json'),collection=pin(folder/'capture-collection.json'),
        transfer=pin(BASE/'capture-transfer.json'),terminal_owners=read(folder/'evidence/initial/collection.json')['identities']+receipt['identities'],reviewer=pin(Path(__file__)),
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),modes=[dict(mode=r['mode'],owned=r['result']['owned_count'],maps=r['result']['retained_maps']) for r in modes],resources=resources)))


if __name__=='__main__':{'build':build,'capture':capture}[sys.argv[1]]()
