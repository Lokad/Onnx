"""Review the consumer-only build and preserve the ownership prediction verdict."""
import base64
import json
from pathlib import Path
import sys
from run import BASE,ROOT,TOOLS,PRELUDE,pin,read,write,ssh,prepared,references


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
    expected=['sdk-version','consumer-restore','consumer-build'] if kind=='build' else ['ownership']
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
    assert set(built['runtime'])==set(spec['runtime_files'])|{'WeightOwnershipProbe.dll','WeightOwnershipProbe.deps.json','WeightOwnershipProbe.runtimeconfig.json'}
    return folder,spec,receipt,state,built,resources


def build():
    assert not (BASE/'build-review.json').exists()
    folder,spec,receipt,state,built,resources=collected('build')
    assert (folder/'logs/sdk-version.stdout').read_text().strip()=='10.0.204'
    for name in ['consumer-restore','consumer-build']:
        text=(folder/'logs'/(name+'.stdout')).read_text()+(folder/'logs'/(name+'.stderr')).read_text()
        assert ': warning ' not in text and ': error ' not in text
    text=(folder/'logs/consumer-build.stdout').read_text()
    assert '0 Warning(s)' in text and '0 Error(s)' in text
    source=(folder/'source/Program.cs').read_text()
    assert source.count('owner.Transcribe(')==1 and source.index('owner.Transcribe(')<source.index('RemoveBindings(encoder, weights)')
    assert 'GC.KeepAlive(owner); GC.KeepAlive(encoder); GC.KeepAlive(contexts);' in source
    for method in ['Describe','OtherInitializers','CheckHeld','RemoveBindings','RemoveMapEntries','CollectAndObserve']:
        assert '[MethodImpl(MethodImplOptions.NoInlining)]' in source[:source.index(' '+method+'(')][-130:]
    result=dict(passed=True,product_rebuilt=False,zero_added_warnings=True,built=pin(folder/'built.json'),
        collection=pin(folder/'build-collection.json'),transfer=pin(BASE/'build-transfer.json'),
        source=pin(folder/'source/Program.cs'),resources=resources,product=spec['product'],
        isolated_evidence=spec['isolated_evidence'],release_admitted=False,reviewer=pin(Path(__file__)))
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
    result=read(folder/'probe/result.json')
    assert result['passed'] and result['runtime']=='.NET 10.0.8' and result['affinity']==4 and result['processor_count']==1
    for key in ['public_request_passed','held_outputs_unchanged','pcm_unchanged','no_inference_after_removal',
                'contexts_live_and_shared','forced_gc_unscored','no_product_change','no_application_score','no_native_runtime']:
        assert result[key],key
    assert result['core_sha256']==spec['product']['Lokad.Onnx.dll']['sha256']
    assert result['data_sha256']==spec['product']['Lokad.Onnx.Data.dll']['sha256']
    assert result['runner_sha256']==built['runtime']['WeightOwnershipProbe.dll']['sha256']
    assert result['spec_sha256']==pin(folder/'spec.json')['sha256']
    weights=read(folder/'probe/weights.json');held=read(folder/'probe/retained.json')
    assert len(weights)==result['weight_count']==96 and len({w['Name'] for w in weights})==96
    assert sum(w['Bytes'] for w in weights)==result['original_bytes']==1610612736
    expected={w['name']:w for w in spec['weights']}
    for row in weights:
        value=expected[row['Name']]
        assert row['Shape']==value['dims'] and row['Cached']==value['mapped']
        assert row['Bytes']==row['Shape'][0]*row['Shape'][1]*4 and len(row['Hash'])==64
    assert sum(w['Cached'] for w in weights)==result['mapped_originals']==9
    assert sum(w['PackedHash'] is not None for w in held)==result['retained_packed_initializers']==37
    assert len(held)==result['non_target_initializers'] and result['remaining_map_entries']==28
    assert len({w['Name'] for w in held})==len(held) and not set(expected)&{w['Name'] for w in held}
    snapshots=[read(folder/'probe'/name) for name in ['original.json','bindings.json','maps.json']]
    assert result['snapshots']==snapshots
    assert [s['Phase'] for s in snapshots]==['original','without-original-bindings','without-original-map-entries']
    for snapshot in snapshots:
        assert [w['Name'] for w in snapshot['Weights']]==[w['Name'] for w in weights]
        assert all(w['Cached']==expected[w['Name']]['mapped'] and w['Bytes']==16777216 for w in snapshot['Weights'])
        assert all(type(v) is int and v>=0 for v in snapshot['Memory'].values())
    roots=(all(w['ArrayAlive'] and w['TensorAlive'] for w in snapshots[0]['Weights'])
        and all(w['ArrayAlive']==w['Cached'] and w['TensorAlive']==w['Cached'] for w in snapshots[1]['Weights'])
        and all(not w['ArrayAlive'] and not w['TensorAlive'] for w in snapshots[2]['Weights']))
    released=snapshots[0]['Memory']['Managed']-snapshots[2]['Memory']['Managed']
    assert released==result['managed_bytes_released']
    memory=released>=1610612736-16*1024**2
    assert result['root_prediction']==roots and result['memory_prediction']==memory and result['prediction_passed']==(roots and memory)
    request=read(folder/'probe/request.json');actual=request['result'];expected_result=spec['case']['expected']
    names={'Text':'text','TokenIds':'token_ids','FrameIndices':'frame_indices','DurationFrames':'duration_frames',
           'EncodedFrames':'encoded_frames','DecoderCalls':'decoder_calls'}
    assert request['passed'] and request['no_timing_score'] and request['name']==spec['case']['name']
    assert request['input_sha256']==spec['case']['raw_sha256']
    assert all(actual[a]==expected_result[b] for a,b in names.items()) and actual['StopReason']==0 and expected_result['stop_reason']=='EndOfAudio'
    analysis=dict(passed=True,prediction_passed=roots and memory,diagnostic_only=True,release_admitted=False,
        no_product_candidate_selected=True,no_application_score=True,original_bytes=1610612736,
        live_arrays=[sum(w['ArrayAlive'] for w in s['Weights']) for s in snapshots],
        live_tensors=[sum(w['TensorAlive'] for w in s['Weights']) for s in snapshots],
        managed_bytes_released=released,snapshots=snapshots,resources=resources,
        public_request=pin(folder/'probe/request.json'),weights=pin(folder/'probe/weights.json'),
        retained=pin(folder/'probe/retained.json'),source=pin(folder/'source/Program.cs'),
        forced_gc_unscored=True,modified_graph_never_executed=True,
        operating_system_release_not_inferred=True,replacement_representation_not_tested=True,
        failed_release_controls=spec['failed_release_controls'])
    write(BASE/'analysis.json',analysis)
    write(BASE/'closed.json',dict(passed=True,prediction_passed=roots and memory,analysis=pin(BASE/'analysis.json'),
        build_review=pin(BASE/'build-review.json'),collection=pin(folder/'capture-collection.json'),
        transfer=pin(BASE/'capture-transfer.json'),terminal_owners=receipt['identities'],reviewer=pin(Path(__file__)),
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),prediction_passed=roots and memory,
        live_arrays=analysis['live_arrays'],managed_bytes_released=released,resources=resources)))


if __name__=='__main__':
    {'build':build,'capture':capture}[sys.argv[1]]()
