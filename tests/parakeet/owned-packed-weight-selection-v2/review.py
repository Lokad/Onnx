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
    expected=['sdk-version','consumer-restore','consumer-build'] if kind=='build' else ['census-512']
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
    assert (folder/'logs/sdk-version.stdout').read_text().strip()=='10.0.204'
    for name in ['consumer-restore','consumer-build']:
        text=(folder/'logs'/(name+'.stdout')).read_text()+(folder/'logs'/(name+'.stderr')).read_text()
        assert ': warning ' not in text and ': error ' not in text
    text=(folder/'logs/consumer-build.stdout').read_text()
    assert '0 Warning(s)' in text and '0 Error(s)' in text
    source=(folder/'source/Program.cs').read_text()
    assert 'owner.Transcribe(' not in source and 'GC.Collect(' not in source
    assert 'RemoveBindings' not in source and 'RemoveMapEntries' not in source
    assert 'PrepareOwnedMatMulWeights' not in source
    result=dict(passed=True,product_rebuilt=False,zero_added_warnings=True,built=pin(folder/'built.json'),
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


def capture():
    assert not (BASE/'closed.json').exists()
    folder,spec,receipt,state,built,resources=collected('capture')
    result=read(folder/'probe/512/result.json');selection=read(folder/'probe/512/selection.json')
    assert result['passed'] and result['diagnostic_only'] and result['no_inference'] and result['mode']=='512'
    assert result['runtime']=='.NET 10.0.8' and result['affinity']==4 and result['processor_count']==1
    assert result['core_sha256']==spec['product']['Lokad.Onnx.dll']['sha256']
    assert result['data_sha256']==spec['product']['Lokad.Onnx.Data.dll']['sha256']
    assert result['runner_sha256']==built['runtime']['OwnedWeightCensus.dll']['sha256']
    assert result['spec_sha256']==pin(folder/'spec.json')['sha256']
    assert len(selection['rows'])==96 and len({r['name'] for r in selection['rows']})==96
    expected={r['Name']:r for r in spec['weights']}
    assert {r['name'] for r in selection['rows']}==set(expected)
    for row in selection['rows']:
        assert row['dimensions']==expected[row['name']]['Shape']
        assert row['kind'] in ['Lokad.Onnx.OwnedPackedTensor','Lokad.Onnx.DenseTensor`1[System.Single]']
        assert row['consumers']
    assert selection['initializer_count']==649 and selection['map_count']==37
    assert selection['owned_count']==sum(r['kind']=='Lokad.Onnx.OwnedPackedTensor' for r in selection['rows'])
    assert selection['owned_bytes']==selection['owned_count']*16777216
    analysis=dict(passed=True,diagnostic_only=True,no_inference=True,selection=selection,result=result,
        product=spec['product'],resources=resources,build_review=pin(BASE/'build-review.json'),release_admitted=False)
    write(BASE/'analysis.json',analysis)
    write(BASE/'closed.json',dict(passed=True,analysis=pin(BASE/'analysis.json'),collection=pin(folder/'capture-collection.json'),
        transfer=pin(BASE/'capture-transfer.json'),terminal_owners=receipt['identities'],reviewer=pin(Path(__file__)),
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),owned_count=selection['owned_count'],rejected=selection['rejected'],
        captured=selection['captured_count'],nested=selection['nested'],resources=resources)))


if __name__=='__main__':{'build':build,'capture':capture}[sys.argv[1]]()
